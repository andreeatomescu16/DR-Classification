#!/usr/bin/env python3
"""Build RunPod master manifest and grouped folds from Kaggle dataset."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


VALID_EXTS = {".jpg", ".jpeg", ".png"}
VALID_LABELS = {"0", "1", "2", "3", "4"}


def find_augmented_root(raw_root: Path) -> Path:
    matches = sorted([p for p in raw_root.rglob("augmented_resized_V2") if p.is_dir()], key=lambda x: len(str(x)))
    if not matches:
        raise FileNotFoundError(f"Could not find 'augmented_resized_V2' under {raw_root}")
    return matches[0]


def strip_suffix_patterns(stem: str) -> str:
    patterns = [
        r"([_-](aug|a)\d+)$",
        r"([_-](flip|fliph|flipv|hflip|vflip))$",
        r"([_-](rot|rotate|r)\d+)$",
        r"([_-](zoom|crop|shift|translate)\d*)$",
        r"([_-](blur|noise|clahe|gamma|bright|contrast)\d*)$",
        r"([_-](copy|dup|variant)\d*)$",
        r"([_-](ver|v)\d+)$",
    ]
    out = stem.lower()
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new_out = re.sub(pat, "", out)
            if new_out != out:
                out = new_out
                changed = True
    out = re.sub(r"[_-]+$", "", out)
    return out if out else stem.lower()


def derive_group_id(image_path: Path) -> str:
    return strip_suffix_patterns(image_path.stem)


def build_master_df(aug_root: Path, folds_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for split_source in ("train", "val", "test"):
        split_dir = aug_root / split_source
        if not split_dir.exists():
            print(f"[warn] missing split folder: {split_dir}")
            continue
        for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()], key=lambda x: x.name):
            if class_dir.name not in VALID_LABELS:
                continue
            label = int(class_dir.name)
            for img in class_dir.rglob("*"):
                if not img.is_file() or img.suffix.lower() not in VALID_EXTS:
                    continue
                rel = os.path.relpath(img, start=folds_dir).replace("\\", "/")
                records.append(
                    {
                        "image_path": rel,
                        "label": label,
                        "split_source": split_source,
                        "group_id": derive_group_id(img),
                    }
                )
    if not records:
        raise RuntimeError("No images discovered in augmented_resized_V2 train/val/test.")
    return pd.DataFrame.from_records(records)


def print_dist(df: pd.DataFrame, key: str, title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    counts = df.groupby(key).size().sort_index()
    for k, v in counts.items():
        print(f"{k}: {v} ({100.0 * v / len(df):.2f}%)")


def print_group_examples(df: pd.DataFrame, n: int = 6) -> None:
    g = df.groupby("group_id")["image_path"].apply(list)
    multi = g[g.apply(len) > 1].sort_values(key=lambda s: s.apply(len), ascending=False)
    print("\nGrouping diagnostics")
    print("-------------------")
    print(f"unique_images={len(df)}")
    print(f"unique_groups={df['group_id'].nunique()}")
    print(f"groups_with_multiple_images={len(multi)}")
    if len(multi) == 0:
        print("[warn] No multi-image groups detected from filename heuristic.")
        return
    print("Sample grouped filenames:")
    shown = 0
    for gid, paths in multi.items():
        print(f"- group_id={gid} count={len(paths)}")
        for p in paths[:3]:
            print(f"    {p}")
        shown += 1
        if shown >= n:
            break


def generate_folds(master: pd.DataFrame, folds_dir: Path, n_splits: int, seed: int) -> None:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = master.index.values
    y = master["label"].values
    groups = master["group_id"].values

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_df = master.copy()
        fold_df["fold"] = fold
        fold_df["is_train"] = False
        fold_df["is_val"] = False
        fold_df.loc[tr_idx, "is_train"] = True
        fold_df.loc[va_idx, "is_val"] = True
        fold_df["split"] = fold_df.apply(
            lambda r: "train" if r["is_train"] else ("val" if r["is_val"] else "ignore"),
            axis=1,
        )
        fold_df["is_valid"] = fold_df["split"].isin(["train", "val"])

        # Keep original test set held out in every fold.
        test_mask = fold_df["split_source"] == "test"
        fold_df.loc[test_mask, "is_train"] = False
        fold_df.loc[test_mask, "is_val"] = False
        fold_df.loc[test_mask, "split"] = "test"
        fold_df.loc[test_mask, "is_valid"] = True

        cols = [
            "image_path",
            "label",
            "fold",
            "is_train",
            "is_val",
            "split",
            "is_valid",
            "group_id",
            "split_source",
        ]
        out = folds_dir / f"fold{fold}.csv"
        fold_df[cols].to_csv(out, index=False)
        print(f"[ok] wrote {out}")


def leakage_stats(fold_df: pd.DataFrame) -> Tuple[int, int, int]:
    tr = set(fold_df.loc[fold_df["split"] == "train", "group_id"].unique().tolist())
    va = set(fold_df.loc[fold_df["split"] == "val", "group_id"].unique().tolist())
    overlap = tr.intersection(va)
    return len(overlap), len(tr), len(va)


def batch_load_test(fold_csv: Path, img_size: int, batch_size: int) -> None:
    from torch.utils.data import DataLoader
    from drlib.datasets import DRDataset
    from drlib.transforms import get_train_tf

    ds = DRDataset(str(fold_csv), split="train", tfm=get_train_tf(size=img_size, remove_borders=True))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    xb, yb = next(iter(dl))
    print(f"[ok] one batch load: images={tuple(xb.shape)} labels={tuple(yb.shape)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default="/workspace/data/raw")
    ap.add_argument("--folds_dir", default="/workspace/data/folds")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    folds_dir = Path(args.folds_dir).resolve()
    folds_dir.mkdir(parents=True, exist_ok=True)

    aug_root = find_augmented_root(raw_root)
    print(f"[ok] dataset root: {aug_root}")

    master = build_master_df(aug_root, folds_dir)
    master = master.sort_values(["split_source", "label", "image_path"]).reset_index(drop=True)
    master_csv = folds_dir / "master.csv"
    master.to_csv(master_csv, index=False)
    print(f"[ok] wrote {master_csv}")

    print(f"\nTotal image count discovered: {len(master)}")
    print_dist(master, "label", "Per-class distribution")
    print_dist(master, "split_source", "Split-source distribution")
    print_group_examples(master)

    generate_folds(master, folds_dir, n_splits=args.n_splits, seed=args.seed)

    print("\nPer-fold distributions + leakage checks")
    print("--------------------------------------")
    for fold in range(args.n_splits):
        fp = folds_dir / f"fold{fold}.csv"
        df = pd.read_csv(fp)
        tab = df.groupby(["split", "label"]).size().unstack(fill_value=0)
        print(f"\nfold{fold}.csv:")
        print(tab.to_string())
        leak, ntr, nva = leakage_stats(df)
        if leak > 0:
            raise RuntimeError(f"group leakage in fold{fold}: {leak} overlapping group_ids")
        print(f"[ok] fold{fold} leakage check passed (train_groups={ntr}, val_groups={nva})")

    batch_load_test(folds_dir / "fold0.csv", img_size=args.img_size, batch_size=args.batch_size)

    print("\nHeld-out policy:")
    print("- split_source=test remains a fixed held-out test set in every fold CSV.")
    print("- Use train/val rows for model selection and report final metrics on test rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
