"""
RetinaGrade — FastAPI backend for Diabetic Retinopathy classification.

Startup:
    python api.py

Endpoints:
    POST /predict   — Upload fundus image → grade, label, probabilities, grad-cam
    GET  /health    — Liveness check
"""

import base64
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Ensure drlib is importable when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drlib.explainability import GradCAM, find_target_layer  # noqa: E402
from drlib.train import DRModule                              # noqa: E402
from drlib.transforms import get_val_tf                      # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

CKPT_DIR = Path("/Users/Andreea/lightning_logs/version_34/checkpoints")

LABELS: List[str] = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Mutable global state (populated once at startup) ─────────────────────────

_state: dict = {
    "module":     None,   # DRModule (Lightning wrapper)
    "model":      None,   # underlying nn.Module (HybridCoAtMini etc.)
    "gcam":       None,   # GradCAM helper (reused; created once)
    "ckpt_path":  None,   # str
    "model_name": None,   # str
    "img_size":   None,   # int
}

# ── Helper: checkpoint discovery ─────────────────────────────────────────────

def _find_checkpoint(ckpt_dir: Path) -> Path:
    """Return best*.ckpt if present, else most-recently-modified .ckpt."""
    candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    best = [c for c in candidates if c.stem.lower().startswith("best")]
    if best:
        return sorted(best, key=lambda p: p.stat().st_mtime)[-1]
    return max(candidates, key=lambda p: p.stat().st_mtime)

# ── Helper: GradCAM initialisation ───────────────────────────────────────────

def _init_gcam(model_nn: torch.nn.Module, model_name: str) -> Optional[GradCAM]:
    """
    Build a GradCAM helper for the model.

    For hybrid models we target stage2.3 (DepthwiseConvBlock, 14×14 at 224px
    input), which is the last CNN stage before the transformer blocks.
    For timm models we let find_target_layer() auto-detect the right layer.
    Returns None if initialisation fails (GradCAM is optional).
    """
    try:
        if "hybrid" in model_name.lower():
            layer_name = "stage2.3"
        else:
            layer_name = find_target_layer(model_nn, model_name)
        print(f"[RetinaGrade] GradCAM target layer: '{layer_name}'")
        return GradCAM(model_nn, layer_name)
    except Exception as exc:
        print(f"[RetinaGrade] GradCAM init skipped — {exc}")
        return None

# ── Lifespan: load model once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = _find_checkpoint(CKPT_DIR)
    print(f"[RetinaGrade] Loading checkpoint: {ckpt}")

    module: DRModule = DRModule.load_from_checkpoint(str(ckpt), map_location="cpu")
    module.eval()
    module.model.eval()

    # Read config that was baked into the checkpoint by save_hyperparameters()
    model_name = str(getattr(module.hparams, "model_name", "unknown"))
    img_size   = int(getattr(module.hparams, "img_size",   224))

    print(f"[RetinaGrade] Model : {model_name}")
    print(f"[RetinaGrade] Input : {img_size}×{img_size}")

    _state["module"]     = module
    _state["model"]      = module.model
    _state["ckpt_path"]  = str(ckpt)
    _state["model_name"] = model_name
    _state["img_size"]   = img_size
    _state["gcam"]       = _init_gcam(module.model, model_name)

    yield
    # nothing to clean up on shutdown

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="RetinaGrade API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # needed for file:// access from the browser
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check — confirms the server is up and which model is loaded."""
    return {
        "status":     "ok",
        "model":      _state["model_name"],
        "checkpoint": _state["ckpt_path"],
        "img_size":   _state["img_size"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classify a fundus image.

    Accepts multipart/form-data with field 'file' (JPG or PNG).

    Returns:
        grade          int        0–4
        label          str        e.g. "Moderate NPDR"
        probabilities  float[5]   softmax scores, sum to 1.0
        gradcam_image  str|null   base64-encoded PNG overlay (null if unavailable)
    """

    # ── 1. Read and decode the uploaded image ────────────────────────────────
    raw_bytes = await file.read()
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Cannot decode image. Please upload a valid JPG or PNG file."
        )
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 2. Preprocess (border removal + resize + normalise) ──────────────────
    img_size: int = _state["img_size"]
    tfm = get_val_tf(size=img_size, remove_borders=True)
    tensor: torch.Tensor = tfm(image=img_rgb)["image"].unsqueeze(0)  # (1,3,H,W)

    # ── 3. Forward pass — probabilities ──────────────────────────────────────
    model: torch.nn.Module = _state["model"]
    with torch.no_grad():
        logits = model(tensor)
        probs_t = torch.softmax(logits, dim=1)[0]   # (5,)

    probs: List[float] = probs_t.cpu().numpy().tolist()
    grade: int = int(np.argmax(probs))

    # ── 4. Grad-CAM overlay (optional — returns null on any failure) ─────────
    gradcam_b64: Optional[str] = None
    gcam: Optional[GradCAM] = _state["gcam"]

    if gcam is not None:
        try:
            # Fresh tensor with grad enabled (detached from prior no_grad graph)
            t_grad = tensor.clone().detach().requires_grad_(True)
            cam = gcam.generate_cam(t_grad, target_class=grade)  # (H', W') float32

            # Denormalise the preprocessed tensor back to uint8 RGB for overlay
            img_np = tensor[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 3) float32
            img_vis = np.clip(
                (img_np * _IMAGENET_STD + _IMAGENET_MEAN) * 255.0,
                0, 255
            ).astype(np.uint8)

            overlay_rgb = gcam.overlay_heatmap(img_vis, cam, alpha=0.45)

            # Encode as PNG → base64 string
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".png", overlay_bgr)
            if ok:
                gradcam_b64 = base64.b64encode(buf).decode("utf-8")

        except Exception as exc:
            print(f"[RetinaGrade] GradCAM inference failed: {exc}")
        finally:
            # Always clean up accumulated gradients
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass

    return {
        "grade":         grade,
        "label":         LABELS[grade],
        "probabilities": probs,
        "gradcam_image": gradcam_b64,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
