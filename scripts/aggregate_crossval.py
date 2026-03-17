#!/usr/bin/env python3
"""
Aggregate 5-fold cross-validation evaluation metrics.

Expected layout:
  <results_dir>/fold0/eval/metrics.csv
  ...
  <results_dir>/fold4/eval/metrics.csv

Writes:
  <results_dir>/summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd


METRICS = [
    "qwk",
    "accuracy",
    "macro_f1",
    "roc_auc_ovr",
    "f1_class_0",
    "f1_class_1",
    "f1_class_2",
    "f1_class_3",
    "f1_class_4",
]


def _read_fold_metrics(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty metrics.csv: {path}")
    row = df.iloc[0]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate cross-val metrics across folds")
    ap.add_argument("--results_dir", default="experiments/focal_crossval", help="Base directory for fold*/eval/metrics.csv")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"[error] results_dir not found: {results_dir}")

    fold_rows = []
    missing = []

    for fold in range(5):
        metrics_path = results_dir / f"fold{fold}" / "eval" / "metrics.csv"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue

        row = _read_fold_metrics(metrics_path)
        row = row.copy()
        row["fold"] = fold
        fold_rows.append(row)

    if not fold_rows:
        raise SystemExit(f"[error] No metrics.csv files found under: {results_dir}")

    df = pd.DataFrame(fold_rows).set_index("fold").sort_index()

    present_metrics = [m for m in METRICS if m in df.columns]
    if not present_metrics:
        raise SystemExit("[error] None of the expected metric columns were found in metrics.csv files.")

    mean = df[present_metrics].mean(axis=0, numeric_only=True)
    std = df[present_metrics].std(axis=0, ddof=1, numeric_only=True)

    summary = pd.DataFrame(
        {
            "metric": present_metrics,
            "mean": [float(mean[m]) for m in present_metrics],
            "std": [float(std[m]) for m in present_metrics],
        }
    )

    # Pretty print
    print("\nCross-validation summary (mean ± std)")
    print("-" * 46)
    for _, r in summary.iterrows():
        print(f"{r['metric']:<14} {r['mean']:.4f} ± {r['std']:.4f}")

    if missing:
        print("\n[warn] Missing folds:")
        for p in missing:
            print(f"  - {p}")

    out_path = results_dir / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

