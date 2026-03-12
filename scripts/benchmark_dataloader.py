#!/usr/bin/env python3
"""
Benchmark script for DR dataloader throughput (no model).

Goals:
- Measure pure data pipeline speed (cv2.imread + Albumentations) in isolation.
- Compare train vs val pipelines, different num_workers, and augmentation strength.
- Help identify CPU bottlenecks (border removal, CLAHE, OpticalDistortion, etc.).

Typical usage on RunPod (from repo root):
    python scripts/benchmark_dataloader.py \
        --fold_csv /workspace/data/folds/fold0.csv \
        --split train \
        --img_size 224 \
        --batch_size 24 \
        --num_workers 8 \
        --n_warmup 10 \
        --n_batches 100 \
        --remove_borders 1 \
        --strong_aug 1

To compare train vs val:
    python scripts/benchmark_dataloader.py --split train ...
    python scripts/benchmark_dataloader.py --split val ...

To compare with/without expensive preprocessing:
    python scripts/benchmark_dataloader.py --split train --remove_borders 1 --strong_aug 1 ...
    python scripts/benchmark_dataloader.py --split train --remove_borders 0 --strong_aug 0 ...
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure local drlib is importable when called as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drlib.datasets import DRDataset
from drlib.transforms import get_train_tf, get_val_tf


def make_dataset_and_loader(
    fold_csv: Path,
    split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    remove_borders: bool,
    strong_aug: bool,
    data_root: str | None,
) -> DataLoader:
    """Construct DRDataset + DataLoader matching training settings."""
    if split == "train":
        tfm = get_train_tf(size=img_size, remove_borders=remove_borders, strong_aug=strong_aug)
    else:
        tfm = get_val_tf(size=img_size, remove_borders=remove_borders)

    ds = DRDataset(
        csv_path=str(fold_csv),
        split=split,
        tfm=tfm,
        data_root=data_root,
    )

    # Match training loader defaults (pin_memory, persistent_workers, etc.)
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=(split == "train"),
    )
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4

    return DataLoader(ds, **dl_kwargs)


def benchmark_loader(
    loader: DataLoader,
    n_warmup: int,
    n_batches: int,
) -> dict:
    """Iterate over the loader and measure throughput in samples/sec."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Num workers: {loader.num_workers}")
    print(f"Batch size : {loader.batch_size}")

    # Warmup: populate worker processes, OS caches, etc.
    print(f"\nWarmup for {n_warmup} batches...")
    it = iter(loader)
    n_warmup_done = 0
    t0 = time.perf_counter()
    while n_warmup_done < n_warmup:
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        # Move to device to simulate real training input path,
        # but do not run any model.
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        n_warmup_done += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"Warmup time: {t1 - t0:.2f} s")

    # Timed loop
    print(f"\nBenchmarking {n_batches} batches...")
    num_samples = 0
    t_start = time.perf_counter()
    it = iter(loader)
    n_done = 0
    batch_times = []
    while n_done < n_batches:
        t_b0 = time.perf_counter()
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_b1 = time.perf_counter()
        batch_times.append(t_b1 - t_b0)
        num_samples += xb.shape[0]
        n_done += 1
        if n_done % 10 == 0 or n_done == n_batches:
            print(f"  Batch {n_done:4d}/{n_batches}: {batch_times[-1]*1000:.1f} ms")

    t_end = time.perf_counter()
    total_time = t_end - t_start

    throughput = num_samples / total_time if total_time > 0 else 0.0
    batch_times_arr = np.array(batch_times, dtype=np.float64)

    print("\n=== Results ===")
    print(f"Total time     : {total_time:.2f} s for {n_batches} batches")
    print(f"Samples loaded : {num_samples}")
    print(f"Throughput     : {throughput:.1f} samples/sec")
    print(
        f"Batch time     : mean={batch_times_arr.mean()*1000:.1f} ms, "
        f"p50={np.percentile(batch_times_arr, 50)*1000:.1f} ms, "
        f"p90={np.percentile(batch_times_arr, 90)*1000:.1f} ms"
    )

    return {
        "num_batches": n_batches,
        "num_samples": num_samples,
        "total_time_s": total_time,
        "throughput_samples_per_s": throughput,
        "batch_time_ms_mean": float(batch_times_arr.mean() * 1000.0),
        "batch_time_ms_p50": float(np.percentile(batch_times_arr, 50) * 1000.0),
        "batch_time_ms_p90": float(np.percentile(batch_times_arr, 90) * 1000.0),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Benchmark DR dataloader throughput (no model).")
    ap.add_argument("--fold_csv", required=True, help="Path to fold CSV (e.g. /workspace/data/folds/fold0.csv)")
    ap.add_argument("--split", choices=["train", "val"], default="train", help="Data split to benchmark")
    ap.add_argument("--img_size", type=int, default=224, help="Target image size")
    ap.add_argument("--batch_size", type=int, default=24, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    ap.add_argument("--remove_borders", type=int, default=1, help="1 = use RemoveBlackBorders, 0 = skip")
    ap.add_argument("--strong_aug", type=int, default=1, help="1 = strong train aug, 0 = light (train only)")
    ap.add_argument("--n_warmup", type=int, default=10, help="Warmup batches (not timed)")
    ap.add_argument("--n_batches", type=int, default=50, help="Timed batches")
    ap.add_argument(
        "--data_root",
        default=None,
        help="Optional override for image root (e.g. /tmp/localdata for NVMe copy).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    fold_csv = Path(args.fold_csv)
    if not fold_csv.exists():
        raise FileNotFoundError(f"fold_csv not found: {fold_csv}")

    print("==============================================")
    print("DR dataloader throughput benchmark")
    print("==============================================")
    print(f"fold_csv      : {fold_csv}")
    print(f"split         : {args.split}")
    print(f"img_size      : {args.img_size}")
    print(f"batch_size    : {args.batch_size}")
    print(f"num_workers   : {args.num_workers}")
    print(f"remove_borders: {bool(args.remove_borders)}")
    print(f"strong_aug    : {bool(args.strong_aug)}")
    print(f"data_root     : {args.data_root or '(CSV paths as-is)'}")
    print(f"n_warmup      : {args.n_warmup}")
    print(f"n_batches     : {args.n_batches}")

    loader = make_dataset_and_loader(
        fold_csv=fold_csv,
        split=args.split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        remove_borders=bool(args.remove_borders),
        strong_aug=bool(args.strong_aug),
        data_root=args.data_root,
    )

    _ = benchmark_loader(loader, n_warmup=args.n_warmup, n_batches=args.n_batches)


if __name__ == "__main__":
    main()

