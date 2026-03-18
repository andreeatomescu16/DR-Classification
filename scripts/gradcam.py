#!/usr/bin/env python3
"""
Grad-CAM visualization script for DR classification models.

Generates per-image 3-panel figures:
  1) Original (denormalized RGB)
  2) Grad-CAM heatmap
  3) Overlay (alpha-blended)

Also generates a thesis-friendly summary grid (5 rows × N cols).

Constraints:
- No external Grad-CAM libraries.
- Load model via DRModule.load_from_checkpoint (same as scripts/evaluate.py).
- Use DRDataset + get_val_tf for data loading.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from drlib.datasets import DRDataset
from drlib.transforms import get_val_tf, IMAGENET_MEAN, IMAGENET_STD


CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) tensor, ImageNet-normalized.
    returns: (3,H,W) in [0,1]
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(3, 1, 1)
    y = x * std + mean
    return torch.clamp(y, 0.0, 1.0)


def tensor_to_uint8_rgb(x01: torch.Tensor) -> np.ndarray:
    """
    x01: (3,H,W) tensor in [0,1]
    returns: (H,W,3) uint8 RGB
    """
    x_np = (x01.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return x_np


def resolve_target_module(model: torch.nn.Module, target_layer: str) -> Tuple[str, torch.nn.Module]:
    """
    Select a module by name from model.named_modules().
    - First tries exact match
    - Then tries suffix match (name.endswith(target_layer))
    - Then tries substring match (target_layer in name)
    """
    named = list(model.named_modules())

    for name, module in named:
        if name == target_layer:
            return name, module

    suffix_matches = [(n, m) for n, m in named if n.endswith(target_layer)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"Target layer '{target_layer}' is ambiguous (suffix matches: {[n for n,_ in suffix_matches][:20]} ...). "
            "Pass --target_layer with an exact module name."
        )

    substring_matches = [(n, m) for n, m in named if target_layer in n]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        raise ValueError(
            f"Target layer '{target_layer}' is ambiguous (substring matches: {[n for n,_ in substring_matches][:20]} ...). "
            "Pass --target_layer with an exact module name."
        )

    raise ValueError(f"Target layer '{target_layer}' not found in model.")


class GradCamHook:
    """Capture activations + gradients for a target layer."""

    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._h_fwd = None
        self._h_bwd = None

    def __enter__(self):
        def fwd_hook(_m, _inp, out):
            self.activations = out

        def bwd_hook(_m, _gin, gout):
            if gout and len(gout) > 0:
                self.gradients = gout[0]

        self._h_fwd = self.module.register_forward_hook(fwd_hook)
        # Prefer full backward hook when available (more correct for complex graphs)
        if hasattr(self.module, "register_full_backward_hook"):
            self._h_bwd = self.module.register_full_backward_hook(bwd_hook)
        else:
            self._h_bwd = self.module.register_backward_hook(bwd_hook)  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._h_fwd is not None:
            self._h_fwd.remove()
        if self._h_bwd is not None:
            self._h_bwd.remove()


def compute_gradcam(
    model: torch.nn.Module,
    x: torch.Tensor,
    class_idx: int,
    hook: GradCamHook,
) -> np.ndarray:
    """
    model: callable on x
    x: (1,3,H,W) normalized
    class_idx: int, target class for Grad-CAM
    returns: cam heatmap (H,W) in [0,1]
    """
    model.zero_grad(set_to_none=True) if hasattr(model, "zero_grad") else model.zero_grad()
    logits = model(x)
    score = logits[0, class_idx]
    score.backward(retain_graph=False)

    if hook.activations is None or hook.gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. Check --target_layer.")

    acts = hook.activations
    grads = hook.gradients

    if acts.ndim != 4 or grads.ndim != 4:
        raise RuntimeError(f"Expected 4D activations/grads, got acts={acts.shape}, grads={grads.shape}")

    # Global-average-pool gradients over spatial dims -> weights (B,C,1,1)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=False)  # (B,H,W)
    cam = F.relu(cam)
    cam = cam[0]

    cam_np = cam.detach().float().cpu().numpy()
    cam_np = cam_np - cam_np.min()
    cam_np = cam_np / (cam_np.max() + 1e-8)
    return cam_np


def cam_to_heatmap_rgb(cam01: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    cam01 = cv2.GaussianBlur(cam01, ksize=(0, 0), sigmaX=8)
    cam_resized = cv2.resize(cam01, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    # Fixed blending weights for clearer overlays:
    # 60% original image + 40% heatmap (requested).
    _ = alpha  # keep signature/backward compatibility; weights are fixed intentionally
    return cv2.addWeighted(image_rgb, 0.6, heatmap_rgb, 0.4, 0.0)


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    dataset: DRDataset,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      preds: (N,) int
      confs: (N,) float in [0,1] for predicted class
    """
    model.eval()
    preds: List[int] = []
    confs: List[float] = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        xs = []
        for i in range(start, end):
            x, _y = dataset[i]
            xs.append(x)
        xb = torch.stack(xs, dim=0).to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)
        p = probs.argmax(dim=1)
        c = probs.gather(1, p.view(-1, 1)).squeeze(1)
        preds.extend(p.detach().cpu().tolist())
        confs.extend(c.detach().cpu().tolist())

    return np.array(preds, dtype=np.int64), np.array(confs, dtype=np.float32)


def stratified_sample_indices(
    dataset: DRDataset,
    preds: np.ndarray,
    confs: np.ndarray,
    num_images: int,
    seed: int,
) -> List[int]:
    """
    Pick ~equal number per true class, mixing correct and incorrect where possible.
    """
    rng = np.random.default_rng(seed)

    labels = dataset.df["label"].to_numpy(dtype=np.int64)
    per_class_base = num_images // 5
    remainder = num_images % 5
    per_class_targets = [per_class_base + (1 if i < remainder else 0) for i in range(5)]
    # Default for num_images=25 -> [5,5,5,5,5]

    selected: List[int] = []
    for cls in range(5):
        target_k = per_class_targets[cls]
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0 or target_k <= 0:
            continue

        correct = cls_idx[preds[cls_idx] == cls]
        incorrect = cls_idx[preds[cls_idx] != cls]

        rng.shuffle(correct)
        rng.shuffle(incorrect)

        # Mix: start with incorrect (if any), then correct, alternating.
        mix: List[int] = []
        i = j = 0
        while len(mix) < target_k and (i < len(incorrect) or j < len(correct)):
            if i < len(incorrect):
                mix.append(int(incorrect[i]))
                i += 1
                if len(mix) >= target_k:
                    break
            if j < len(correct):
                mix.append(int(correct[j]))
                j += 1

        # If still short (e.g., all correct or all incorrect), fill from remaining.
        if len(mix) < target_k:
            remaining = [int(x) for x in cls_idx if int(x) not in set(mix)]
            rng.shuffle(remaining)
            mix.extend(remaining[: max(0, target_k - len(mix))])

        selected.extend(mix[:target_k])

    # If we didn't reach num_images (rare), fill from the rest.
    if len(selected) < num_images:
        remaining_all = [i for i in range(len(dataset)) if i not in set(selected)]
        rng.shuffle(remaining_all)
        selected.extend(remaining_all[: num_images - len(selected)])

    return selected[:num_images]


def safe_stem(p: str) -> str:
    s = Path(p).name
    s = s.replace(" ", "_")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for DR model checkpoints")
    ap.add_argument("--checkpoint", required=True, help="Path to Lightning .ckpt")
    ap.add_argument("--data_csv", required=True, help="Fold CSV (e.g., /workspace/data/folds/fold0.csv)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to sample from")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_images", type=int, default=25)
    ap.add_argument("--out_dir", default="gradcam_outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default=None, help="Optional data_root override (same as training)")
    ap.add_argument("--target_layer", default="stage2", help="Target layer name (matched against model.named_modules())")
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha for heatmap")
    args = ap.parse_args()

    set_seed(args.seed)
    device = auto_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model (same approach as scripts/evaluate.py)
    from drlib.train import DRModule

    model = DRModule.load_from_checkpoint(args.checkpoint, strict=False)
    model.eval()
    model.to(device)

    # Print full named module list for target layer confirmation
    print("\n" + "=" * 80)
    print("MODEL NAMED MODULES")
    print("=" * 80)
    for name, module in model.named_modules():
        if name == "":
            continue
        print(name, "->", module.__class__.__name__)
    print("=" * 80 + "\n")

    target_name, target_module = resolve_target_module(model, args.target_layer)
    print(f"[gradcam] Using target_layer='{args.target_layer}' resolved as '{target_name}' ({target_module.__class__.__name__})")

    # Dataset (same preprocessing as evaluation)
    tfm = get_val_tf(args.img_size, remove_borders=True)
    dataset = DRDataset(args.data_csv, split=args.split, tfm=tfm, data_root=args.data_root)
    if len(dataset) == 0:
        raise SystemExit(f"[error] Dataset is empty for split='{args.split}'. Try --split val.")

    # Predict all samples to enable correct/incorrect stratified selection
    print(f"[gradcam] Running predictions over {len(dataset)} samples to select a balanced subset...")
    preds, confs = predict_all(model, dataset, device=device, batch_size=64)

    indices = stratified_sample_indices(dataset, preds, confs, num_images=args.num_images, seed=args.seed)
    print(f"[gradcam] Selected {len(indices)} samples (target={args.num_images}).")

    # Group by true class for summary grid
    by_true: Dict[int, List[int]] = {i: [] for i in range(5)}
    for idx in indices:
        true = int(dataset.df.iloc[idx]["label"])
        by_true[true].append(idx)

    # Summary grid settings
    grid_cols = max(len(by_true[i]) for i in range(5)) if args.num_images > 0 else 0
    grid_cols = max(grid_cols, 1)
    grid_cols = min(grid_cols, 5)  # thesis figure: up to 5 columns

    fig_grid, axes_grid = plt.subplots(5, grid_cols, figsize=(3.2 * grid_cols, 3.2 * 5))
    if grid_cols == 1:
        axes_grid = axes_grid.reshape(5, 1)

    # Generate per-image figures + populate summary grid with overlays
    with GradCamHook(target_module) as hook:
        for cls in range(5):
            cls_indices = by_true[cls][:grid_cols]
            for col in range(grid_cols):
                ax = axes_grid[cls, col]
                ax.axis("off")
                if col >= len(cls_indices):
                    continue

                idx = cls_indices[col]
                x, y = dataset[idx]
                x1 = x.unsqueeze(0).to(device)

                # Forward for pred/conf (no_grad ok)
                with torch.no_grad():
                    logits = model(x1)
                    probs = F.softmax(logits, dim=1)[0]
                    pred = int(probs.argmax().item())
                    conf = float(probs[pred].item())

                # Grad-CAM requires grad
                x1 = x1.requires_grad_(True)
                cam01 = compute_gradcam(model, x1, class_idx=pred, hook=hook)

                # Visual assets
                img_denorm = denormalize_imagenet(x.to(device))
                img_rgb = tensor_to_uint8_rgb(img_denorm)
                heat_rgb = cam_to_heatmap_rgb(cam01, out_h=img_rgb.shape[0], out_w=img_rgb.shape[1])
                overlay = overlay_heatmap(img_rgb, heat_rgb, alpha=float(args.alpha))

                true_name = CLASS_NAMES[int(y)]
                pred_name = CLASS_NAMES[pred]
                correct = (pred == int(y))

                title = f"True: {true_name} | Pred: {pred_name} | Conf: {conf:.1%}"
                title_color = "green" if correct else "red"

                # Per-image 3-panel figure
                fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.5))
                for a in axs:
                    a.axis("off")
                axs[0].imshow(img_rgb)
                axs[0].set_title("Original")
                axs[1].imshow(cam01, cmap="jet")
                axs[1].set_title("Grad-CAM")
                axs[2].imshow(overlay)
                axs[2].set_title("Overlay")
                fig.suptitle(title, color=title_color, fontsize=12)
                fig.tight_layout()

                filename = safe_stem(str(dataset.df.iloc[idx]["image_path"]))
                out_name = f"{int(y)}_{pred}_{filename}.png"
                out_path = out_dir / out_name
                fig.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                # Summary grid uses overlay only
                ax.imshow(overlay)
                ax.set_title(f"{pred} ({conf:.0%})", color=title_color, fontsize=9)

            # row label on first column
            axes_grid[cls, 0].set_ylabel(CLASS_NAMES[cls], rotation=90, fontsize=10)

    fig_grid.suptitle("Grad-CAM Summary (rows = true class)", fontsize=14)
    fig_grid.tight_layout()
    grid_path = out_dir / "gradcam_summary.png"
    fig_grid.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig_grid)

    print(f"[gradcam] Saved individual figures to: {out_dir}")
    print(f"[gradcam] Saved summary grid to: {grid_path}")


if __name__ == "__main__":
    main()

