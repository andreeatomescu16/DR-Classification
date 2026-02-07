#!/usr/bin/env python3
"""
Script pentru procesarea dataset-ului combinat descărcat de pe Kaggle.
Detectează automat structura și procesează EyePACS și APTOS.
"""

import argparse
import sys
from pathlib import Path
import subprocess
import pandas as pd


def find_dataset_directories(dataset_dir):
    """Găsește directoarele EyePACS și APTOS în dataset-ul combinat."""
    eyepacs_root = None
    aptos_root = None
    
    print("🔍 Căutare structură dataset...")
    
    # Caută directoare cu nume care conțin "eyepacs"
    for item in dataset_dir.rglob("*"):
        if item.is_dir():
            name_lower = item.name.lower()
            if "eyepacs" in name_lower or ("eye" in name_lower and "pacs" in name_lower):
                eyepacs_root = item
                print(f"   ✓ Found EyePACS directory: {eyepacs_root}")
                break
    
    # Dacă nu găsește, caută directoare cu imagini care arată ca EyePACS
    if not eyepacs_root:
        for item in dataset_dir.iterdir():
            if item.is_dir():
                jpg_files = list(item.rglob("*.jpg"))
                if len(jpg_files) > 1000:  # EyePACS are multe imagini
                    has_label_dirs = any((item / str(i)).exists() for i in range(5))
                    has_splits = (item / "train").exists() or (item / "val").exists()
                    if has_label_dirs or has_splits:
                        eyepacs_root = item
                        print(f"   ✓ Found potential EyePACS directory: {eyepacs_root}")
                        break
    
    # Caută APTOS
    for item in dataset_dir.rglob("*"):
        if item.is_dir():
            name_lower = item.name.lower()
            if "aptos" in name_lower:
                aptos_root = item
                print(f"   ✓ Found APTOS directory: {aptos_root}")
                break
    
    # Dacă nu găsește, caută directoare cu train_images
    if not aptos_root:
        for item in dataset_dir.rglob("train_images"):
            if item.is_dir() and any(item.glob("*.png")):
                aptos_root = item.parent
                print(f"   ✓ Found potential APTOS directory: {aptos_root}")
                break
    
    return eyepacs_root, aptos_root


def process_eyepacs(eyepacs_root, output_dir):
    """Procesează dataset-ul EyePACS."""
    print("\n📋 Processing EyePACS...")
    
    if not eyepacs_root:
        print("⚠ EyePACS directory not found - skipping")
        return None
    
    # Caută trainLabels.csv
    labels_csv = None
    for csv_file in eyepacs_root.rglob("*.csv"):
        if "label" in csv_file.name.lower() or "train" in csv_file.name.lower():
            labels_csv = csv_file
            print(f"   Found labels CSV: {labels_csv}")
            break
    
    # Găsește root-ul cu imagini
    img_root = eyepacs_root
    if (eyepacs_root / "train").exists():
        img_root = eyepacs_root
    else:
        # Caută primul director cu multe imagini
        for subdir in eyepacs_root.rglob("*"):
            if subdir.is_dir():
                jpg_count = len(list(subdir.glob("*.jpg")))
                if jpg_count > 100:
                    img_root = eyepacs_root
                    break
    
    output_csv = Path(output_dir) / "eyepacs_master.csv"
    
    cmd = [
        sys.executable, "scripts/prepare_eyepacs.py",
        "--img_root", str(img_root),
        "--out_csv", str(output_csv)
    ]
    
    if labels_csv:
        cmd.extend(["--labels_csv", str(labels_csv)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    if output_csv.exists():
        print(f"✓ EyePACS master CSV created: {output_csv}")
        return output_csv
    else:
        print("⚠ EyePACS master CSV not created - check errors above")
        return None


def process_aptos(aptos_root, output_dir):
    """Procesează dataset-ul APTOS."""
    print("\n📋 Processing APTOS...")
    
    if not aptos_root:
        print("⚠ APTOS directory not found - skipping")
        return None
    
    # Caută train.csv și train_images
    train_csv = None
    train_imgdir = None
    
    for csv_file in aptos_root.rglob("train.csv"):
        train_csv = csv_file
        print(f"   Found train CSV: {train_csv}")
        break
    
    for img_dir in aptos_root.rglob("train_images"):
        if img_dir.is_dir():
            train_imgdir = img_dir
            print(f"   Found train images: {train_imgdir}")
            break
    
    if not train_csv or not train_imgdir:
        print("⚠ APTOS train.csv or train_images not found")
        return None
    
    # Caută val.csv și val_images (opțional)
    val_csv = None
    val_imgdir = None
    
    for csv_file in aptos_root.rglob("val.csv"):
        val_csv = csv_file
        break
    
    for img_dir in aptos_root.rglob("val_images"):
        if img_dir.is_dir():
            val_imgdir = img_dir
            break
    
    output_csv = Path(output_dir) / "aptos_master.csv"
    
    if val_csv and val_imgdir:
        cmd = [
            sys.executable, "scripts/prepare_apots.py",
            "--train_csv", str(train_csv),
            "--train_imgdir", str(train_imgdir),
            "--val_csv", str(val_csv),
            "--val_imgdir", str(val_imgdir),
            "--out_csv", str(output_csv)
        ]
    else:
        print("   ⚠ APTOS val split not found, using only train")
        cmd = [
            sys.executable, "scripts/prepare_apots.py",
            "--train_csv", str(train_csv),
            "--train_imgdir", str(train_imgdir),
            "--val_csv", str(train_csv),
            "--val_imgdir", str(train_imgdir),
            "--out_csv", str(output_csv)
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Warnings:", result.stderr)
    
    if output_csv.exists():
        print(f"✓ APTOS master CSV created: {output_csv}")
        return output_csv
    else:
        print("⚠ APTOS master CSV not created - check errors above")
        return None


def main():
    ap = argparse.ArgumentParser(description="Process combined dataset from Kaggle")
    ap.add_argument("--dataset_dir", required=True, help="Directory containing combined dataset")
    ap.add_argument("--output_dir", default="data", help="Output directory for master CSVs")
    
    args = ap.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return 1
    
    print("="*60)
    print("PROCESARE DATASET COMBINAT")
    print("="*60)
    
    # Găsește directoarele
    eyepacs_root, aptos_root = find_dataset_directories(dataset_dir)
    
    # Procesează EyePACS
    eyepacs_csv = process_eyepacs(eyepacs_root, output_dir)
    
    # Procesează APTOS
    aptos_csv = process_aptos(aptos_root, output_dir)
    
    print("\n" + "="*60)
    print("PROCESARE COMPLETĂ")
    print("="*60)
    
    if eyepacs_csv:
        df = pd.read_csv(eyepacs_csv)
        print(f"✓ EyePACS: {len(df)} samples")
    
    if aptos_csv:
        df = pd.read_csv(aptos_csv)
        print(f"✓ APTOS: {len(df)} samples")
    
    if eyepacs_csv and aptos_csv:
        print("\n✅ Ambele dataset-uri procesate cu succes!")
        print("Poți continua cu crearea K-fold splits:")
        print(f"  python scripts/kfold_split.py --masters {eyepacs_csv} {aptos_csv} --out_dir data/folds")
    elif eyepacs_csv or aptos_csv:
        print("\n⚠ Doar un dataset a fost procesat cu succes")
    else:
        print("\n✗ Niciun dataset nu a fost procesat")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
