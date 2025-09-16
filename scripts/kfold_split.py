#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def main():
    ap = argparse.ArgumentParser(description="Create grouped, stratified K-fold splits for DR datasets")
    ap.add_argument("--masters", nargs="+", required=True,
                    help="One or more master CSVs (APTOS/EyePACS)")
    ap.add_argument("--out_dir", default="data/folds",
                    help="Output directory for folds")
    ap.add_argument("--n_splits", type=int, default=5,
                    help="Number of folds (default: 5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    args = ap.parse_args()

    # === Load & merge all masters ===
    dfs = []
    for f in args.masters:
        df = pd.read_csv(f)
        df["source_file"] = Path(f).name  # keep provenance
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # === Clean ===
    df = df[df["is_valid"] == True].copy()
    df = df[(df["image_path"].notnull()) & (df["image_path"] != "")]
    df = df[df["label"].isin([0, 1, 2, 3, 4])]  # drop unlabeled
    df.reset_index(drop=True, inplace=True)

    # required columns
    for c in ["patient_id", "label", "image_path"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === StratifiedGroupKFold ===
    sgkf = StratifiedGroupKFold(n_splits=args.n_splits,
                                shuffle=True, random_state=args.seed)
    X = df.index.values
    y = df["label"].values
    groups = df["patient_id"].astype(str).values

    for fold_id, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_df = df.copy()
        fold_df["split"] = "train"
        fold_df.loc[va_idx, "split"] = "val"

        fold_path = out_dir / f"fold{fold_id}.csv"
        fold_df.to_csv(fold_path, index=False)

        # per-fold summary
        rep = (fold_df.groupby(["split", "label"])
               .size()
               .unstack(fill_value=0)
               .reset_index())
        rep_path = out_dir / f"fold{fold_id}_summary.csv"
        rep.to_csv(rep_path, index=False)

        print(f"[fold {fold_id}] wrote {fold_path} ({len(fold_df)} rows), "
              f"summary -> {rep_path}")

    # optional: combined catalog
    catalog_path = out_dir / "catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f"Wrote combined catalog: {catalog_path} ({len(df)} rows total)")

if __name__ == "__main__":
    main()
