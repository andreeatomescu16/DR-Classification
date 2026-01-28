#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
import pandas as pd
from PIL import Image

def build_index(img_root: Path):
    """Build a case-insensitive index: stem -> full path."""
    idx = {}
    for p in img_root.rglob("*"):
        if p.is_file():
            idx[p.stem.lower()] = p
    return idx

def process_split(split_name, labels_csv, img_dir, dataset_name="aptos2019"):
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    if not {"id_code","diagnosis"}.issubset(df.columns):
        raise ValueError(f"{labels_csv} must have id_code, diagnosis")
    df["id_code"] = df["id_code"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(int)

    index = build_index(Path(img_dir))
    rows = []
    for _, r in df.iterrows():
        stem = str(r["id_code"]).strip()
        label = int(r["diagnosis"])
        p = index.get(stem.lower(), None)

        is_valid, w, h = False, None, None
        if p is not None:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    im.verify()
                is_valid = True
            except Exception:
                is_valid = False

        rows.append({
            "dataset": dataset_name,
            "split": split_name,
            "image_path": str(p) if p else "",
            "id_code": stem,
            "patient_id": stem,  # APTOS has no patient grouping
            "eye": "U",
            "label": label,
            "width": w, "height": h,
            "is_valid": is_valid
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Prepare APTOS master CSV (train+val)")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--train_imgdir", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--val_imgdir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df_train = process_split("train", args.train_csv, args.train_imgdir)
    df_val   = process_split("val", args.val_csv, args.val_imgdir)

    df = pd.concat([df_train, df_val], ignore_index=True)
    n_missing = (df["image_path"] == "").sum()
    if n_missing:
        print(f"WARNING: {n_missing} images missing!", file=sys.stderr)

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} with {len(df)} rows; valid={df['is_valid'].sum()}")

if __name__ == "__main__":
    main()
