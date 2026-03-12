#!/usr/bin/env python3
"""
Quick profiling script: parameter count, GPU memory, and per-step timing.

Usage (RunPod, from repo root):
    python scripts/profile_model.py --model hybrid_coatmini
    python scripts/profile_model.py --model hybrid_coat_small
    python scripts/profile_model.py --model hybrid_coat_small --batch_size 16
    python scripts/profile_model.py --compare  # side-by-side table
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from drlib.models import create_model, _SCRATCH_MODELS

# ── helpers ────────────────────────────────────────────────────────────────


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def bytes_to_mb(b: int) -> float:
    return b / 1024 / 1024


def profile_model(
    model_name: str,
    img_size: int = 224,
    batch_size: int = 24,
    n_warmup: int = 3,
    n_timed: int = 10,
    precision: str = "16-mixed",
    verbose: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Model : {model_name}")
        print(f"  Device: {device}")
        print(f"  Input : {batch_size} × 3 × {img_size} × {img_size}")
        print(f"  Prec  : {precision}")
        print(f"{'='*60}")

    # ── Build model ─────────────────────────────────────────────────────────
    model = create_model(model_name, num_classes=5, pretrained=False).to(device)
    model.train()

    total_params, trainable_params = count_params(model)
    if verbose:
        print(f"  Params (total)    : {total_params:,}  ({total_params/1e6:.3f} M)")
        print(f"  Params (trainable): {trainable_params:,}  ({trainable_params/1e6:.3f} M)")

    # ── Dummy batch ─────────────────────────────────────────────────────────
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, 5, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    use_amp = precision == "16-mixed" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def _step():
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        return loss.item()

    # ── Warmup ──────────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if verbose:
        print(f"\n  Warming up ({n_warmup} steps)...", end=" ", flush=True)
    for _ in range(n_warmup):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    if verbose:
        print("done")

    mem_after_warmup = bytes_to_mb(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    peak_mem = bytes_to_mb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    # ── Timed run ───────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        _step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    step_ms = elapsed / n_timed * 1000
    step_s = step_ms / 1000

    # ── Epoch / cost estimates ───────────────────────────────────────────────
    # fold0 train: 103,560 samples / batch_size steps per epoch
    n_train = 103_560
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    epoch_s = steps_per_epoch * step_s
    epoch_min = epoch_s / 60
    # 40 epochs
    total_h = epoch_s * 40 / 3600
    cost_usd = total_h * 0.65

    if verbose:
        print(f"\n  GPU memory allocated : {mem_after_warmup:.1f} MB")
        print(f"  GPU peak memory      : {peak_mem:.1f} MB")
        total_vram = bytes_to_mb(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0
        if total_vram > 0:
            print(f"  GPU total VRAM       : {total_vram:.0f} MB  ({peak_mem/total_vram*100:.1f}% used at peak)")
        print(f"\n  Step time (fwd+bwd)  : {step_ms:.1f} ms")
        print(f"  Steps per epoch      : {steps_per_epoch:,}  (n_train={n_train:,}, bs={batch_size})")
        print(f"  Estimated epoch time : {epoch_min:.1f} min")
        print(f"  Estimated 40-ep time : {total_h:.2f} h")
        print(f"  Estimated 40-ep cost : ${cost_usd:.2f}  (@ $0.65/h)")

    return {
        "model": model_name,
        "total_params_M": round(total_params / 1e6, 3),
        "trainable_params_M": round(trainable_params / 1e6, 3),
        "step_ms": round(step_ms, 1),
        "gpu_alloc_MB": round(mem_after_warmup, 1),
        "gpu_peak_MB": round(peak_mem, 1),
        "epoch_min": round(epoch_min, 1),
        "total_h_40ep": round(total_h, 2),
        "cost_40ep_usd": round(cost_usd, 2),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Profile one or more DR models")
    ap.add_argument("--model", default="hybrid_coat_small",
                    help="Model name (hybrid_coatmini | hybrid_coat_small | any timm model)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--precision", default="16-mixed",
                    choices=["32-true", "16-mixed"],
                    help="16-mixed = AMP (matches training)")
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_timed", type=int, default=10)
    ap.add_argument("--compare", action="store_true",
                    help="Profile both hybrid_coatmini and hybrid_coat_small side-by-side")
    args = ap.parse_args()

    if args.compare:
        models = ["hybrid_coatmini", "hybrid_coat_small"]
        results = []
        for m in models:
            r = profile_model(
                m,
                img_size=args.img_size,
                batch_size=args.batch_size,
                n_warmup=args.n_warmup,
                n_timed=args.n_timed,
                precision=args.precision,
            )
            results.append(r)

        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        keys = ["model", "total_params_M", "step_ms", "gpu_peak_MB",
                "epoch_min", "total_h_40ep", "cost_40ep_usd"]
        labels = ["Model", "Params(M)", "Step(ms)", "PeakMem(MB)",
                  "Epoch(min)", "40ep(h)", "40ep($)"]
        col_w = 18
        header = "  " + "".join(f"{l:<{col_w}}" for l in labels)
        print(header)
        print("  " + "-" * (col_w * len(labels)))
        for r in results:
            row = "  " + "".join(f"{str(r[k]):<{col_w}}" for k in keys)
            print(row)
        print()
    else:
        profile_model(
            args.model,
            img_size=args.img_size,
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            n_timed=args.n_timed,
            precision=args.precision,
        )


if __name__ == "__main__":
    main()
