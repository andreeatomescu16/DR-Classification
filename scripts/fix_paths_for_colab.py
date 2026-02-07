#!/usr/bin/env python3
"""
Script to fix image paths in CSV files for Google Colab.

This script:
1. Detects if running on Colab
2. Updates absolute paths to relative paths or Colab-compatible paths
3. Creates a new CSV file with corrected paths
"""

import pandas as pd
import os
from pathlib import Path
import argparse


def is_colab():
    """Check if running on Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def find_images_directory(base_path):
    """Find the directory containing images."""
    # Common locations
    possible_dirs = [
        base_path / 'data' / 'images',
        base_path / 'data' / 'train_images',
        base_path / 'images',
        base_path / 'train_images',
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists() and dir_path.is_dir():
            return dir_path
    
    # Search recursively
    for root, dirs, files in os.walk(base_path):
        if 'images' in root.lower() or 'train' in root.lower():
            dir_path = Path(root)
            if any(dir_path.glob('*.png')) or any(dir_path.glob('*.jpg')):
                return dir_path
    
    return None


def fix_path(row, base_path, images_dir=None):
    """Fix a single image path."""
    img_path = row['image_path']
    
    # If already a relative path and exists, keep it
    if Path(img_path).exists():
        return img_path
    
    # If absolute path, try to convert to relative
    if os.path.isabs(img_path):
        # Extract filename
        filename = os.path.basename(img_path)
        
        # Try to find in common locations
        if images_dir and images_dir.exists():
            new_path = images_dir / filename
            if new_path.exists():
                return str(new_path.relative_to(base_path))
        
        # Try relative to base_path
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = base_path / 'data' / 'images' / filename
            if test_path.exists():
                return str(test_path.relative_to(base_path))
            
            test_path = base_path / 'images' / filename
            if test_path.exists():
                return str(test_path.relative_to(base_path))
    
    # If can't find, return original (will fail later with better error)
    return img_path


def fix_csv_paths(input_csv, output_csv=None, base_path=None):
    """Fix paths in CSV file."""
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    print(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'image_path' not in df.columns:
        print("ERROR: 'image_path' column not found in CSV!")
        return False
    
    print(f"Found {len(df)} rows")
    
    # Find images directory
    images_dir = find_images_directory(base_path)
    if images_dir:
        print(f"Found images directory: {images_dir}")
    else:
        print("WARNING: Could not find images directory automatically")
    
    # Fix paths
    print("Fixing paths...")
    fixed_count = 0
    for idx, row in df.iterrows():
        old_path = row['image_path']
        new_path = fix_path(row, base_path, images_dir)
        
        if old_path != new_path:
            df.at[idx, 'image_path'] = new_path
            fixed_count += 1
    
    print(f"Fixed {fixed_count} paths")
    
    # Save
    if output_csv is None:
        output_csv = str(Path(input_csv).with_suffix('.colab.csv'))
    
    df.to_csv(output_csv, index=False)
    print(f"Saved fixed CSV to: {output_csv}")
    
    # Verify
    print("\nVerifying paths...")
    missing = []
    for idx, row in df.head(10).iterrows():  # Check first 10
        path = Path(row['image_path'])
        if not path.exists():
            missing.append(row['image_path'])
    
    if missing:
        print(f"WARNING: {len(missing)} paths in first 10 rows don't exist:")
        for p in missing[:3]:
            print(f"  - {p}")
    else:
        print("✓ All checked paths exist!")
    
    return True


def main():
    ap = argparse.ArgumentParser(description="Fix image paths in CSV for Colab")
    ap.add_argument("input_csv", help="Input CSV file")
    ap.add_argument("--output_csv", default=None, help="Output CSV file (default: input.colab.csv)")
    ap.add_argument("--base_path", default=None, help="Base path for relative paths")
    
    args = ap.parse_args()
    
    if is_colab():
        print("✓ Running on Google Colab")
    else:
        print("⚠ Not running on Colab (but will still fix paths)")
    
    fix_csv_paths(args.input_csv, args.output_csv, args.base_path)


if __name__ == "__main__":
    main()
