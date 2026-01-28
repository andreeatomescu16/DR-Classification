#!/usr/bin/env python3
import argparse, sys, re
from pathlib import Path
import pandas as pd
from PIL import Image

# folder names "0..4" used as labels
LABEL_DIRS = {"0", "1", "2", "3", "4"}
# remove trailing size suffixes like -600 / _768 / -1024
SIZE_SUFFIX_RE = re.compile(r"[-_](\d{3,4})$", re.IGNORECASE)
# parse "<pid>_left" / "<pid>_right"
SIDE_RE = re.compile(r"^(?P<pid>.+?)_(?P<side>left|right)$", re.IGNORECASE)
# detect split name from parents
SPLIT_NAMES = {"train", "val", "valid", "validation", "test"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def normalize_stem(stem: str) -> str:
    """Strip common size suffixes like -600 / _768."""
    m = SIZE_SUFFIX_RE.search(stem)
    return stem[:m.start()] if m else stem

def parse_patient_eye(norm_stem: str):
    """Return (patient_id, eye[L/R/U]) from a normalized stem."""
    m = SIDE_RE.match(norm_stem)
    if not m:
        return norm_stem, "U"
    pid = m.group("pid")
    side = m.group("side").lower()
    return pid, ("L" if side == "left" else "R")

def nearest_label_from_parents(p: Path):
    """Find an ancestor directory named 0..4 and return it as int label."""
    for parent in [p.parent, *p.parents]:
        name = parent.name.strip()
        if name in LABEL_DIRS:
            return int(name)
    return None

def find_source_split(p: Path) -> str:
    """Find a recognizable split name in parents (train/val/valid/validation/test)."""
    for parent in [p.parent, *p.parents]:
        n = parent.name.lower()
        if n in SPLIT_NAMES:
            # normalize valid/validation -> val
            if n in {"valid", "validation"}:
                return "val"
            return n
    return "other"

def build_label_map(labels_csv: Path) -> dict:
    """Map image name (WITHOUT size suffix) -> label from CSV."""
    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    # accept common column pairs
    col_image = None
    for c in ["image", "image_name", "id_code", "filename"]:
        if c in df.columns:
            col_image = c
            break
    col_level = None
    for c in ["level", "diagnosis", "label"]:
        if c in df.columns:
            col_level = c
            break
    if not col_image or not col_level:
        raise ValueError("labels CSV must have (image, level) or (image_name, diagnosis/label) columns")

    df[col_image] = df[col_image].astype(str).str.strip()
    df[col_level] = df[col_level].astype(int)

    # normalize key by stripping size suffix from stem (so '1_left-600' -> '1_left')
    key = df[col_image].apply(lambda x: normalize_stem(Path(x).stem))
    return dict(zip(key, df[col_level]))

def main():
    ap = argparse.ArgumentParser(description="Prepare EyePACS master CSV from a folder tree (and/or labels CSV).")
    ap.add_argument("--img_root", required=True, help="Root folder with EyePACS images (contains train/, val/, etc.)")
    ap.add_argument("--out_csv", required=True, help="Output master CSV path")
    ap.add_argument("--labels_csv", default=None, help="Optional labels CSV (image,level) or (image_name,diagnosis)")
    ap.add_argument("--include_splits", nargs="*", default=["train", "val"],
                    help="Split folders to include (default: train val). If omitted entirely, scans all.")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    if not img_root.exists():
        print(f"ERROR: --img_root not found: {img_root}", file=sys.stderr)
        sys.exit(1)

    label_map = None
    if args.labels_csv:
        label_map = build_label_map(Path(args.labels_csv))

    include = set([s.lower() for s in args.include_splits]) if args.include_splits else None

    # Collect files
    files = []
    for p in img_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            if include:
                parents = {pp.name.lower() for pp in p.parents}
                if not any(s in parents for s in include):
                    continue
            files.append(p)

    if not files:
        print(f"WARNING: No images found under {img_root}. Ensure they are downloaded locally (iCloud: right-click → Download).", file=sys.stderr)

    rows = []
    for f in files:
        stem = f.stem                          # e.g., "1_left-600" or "0eced86c9db8-600"
        norm_stem = normalize_stem(stem)       # -> "1_left" or "0eced86c9db8"
        pid, eye = parse_patient_eye(norm_stem)
        source_split = find_source_split(f)

        # Determine label: CSV map first (if provided), otherwise from folder names 0..4
        label = label_map.get(norm_stem) if label_map else None
        if label is None:
            label = nearest_label_from_parents(f)

        is_valid, w, h = False, None, None
        try:
            with Image.open(f) as im:
                w, h = im.size
                im.verify()
            is_valid = True
        except Exception:
            is_valid = False

        rows.append({
            "dataset": "eyepacs2015",
            "source_split": source_split,   # train/val/test/other (from folder)
            "image_path": str(f),
            "id_code": norm_stem,          # no size suffix
            "patient_id": pid,
            "eye": eye,                    # L/R/U
            "label": label,                # 0..4 (may be None if unlabeled)
            "width": w, "height": h,
            "is_valid": is_valid
        })

    df = pd.DataFrame(rows)
    # Basic warnings & write
    unlabeled = int(df["label"].isna().sum()) if not df.empty else 0
    print(f"Found {len(df)} images. Valid={int(df['is_valid'].sum()) if not df.empty else 0}; unlabeled={unlabeled}")
    if unlabeled:
        print("NOTE: Unlabeled rows remain in the CSV; kfold script will drop rows without labels.", file=sys.stderr)

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

if __name__ == "__main__":
    main()
