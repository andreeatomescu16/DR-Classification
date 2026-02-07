#!/usr/bin/env python3
"""
Fix CSV paths for Google Colab - converts absolute paths to relative paths.

Usage:
    python scripts/fix_csv_paths_colab.py data/folds/fold0.csv --images_dir /path/to/images
"""

import pandas as pd
import argparse
from pathlib import Path
import os


def find_image_by_filename(filename, search_dirs):
    """Find image file by filename in search directories."""
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Try direct match
        direct_path = search_dir / filename
        if direct_path.exists():
            return direct_path
        
        # Try recursive search
        matches = list(search_dir.rglob(filename))
        if matches:
            return matches[0]
    
    return None


def fix_csv_paths(input_csv, output_csv=None, images_dir=None, base_path=None):
    """Fix absolute paths in CSV to relative paths."""
    if base_path is None:
        base_path = Path.cwd()
    else:
        base_path = Path(base_path)
    
    print(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    if 'image_path' not in df.columns:
        print("ERROR: 'image_path' column not found!")
        return False
    
    print(f"Found {len(df)} rows")
    
    # Determine search directories
    search_dirs = []
    if images_dir:
        search_dirs.append(Path(images_dir))
    
    # Add common locations relative to CSV
    csv_path = Path(input_csv)
    csv_dir = csv_path.parent if csv_path.is_file() else csv_path
    search_dirs.extend([
        csv_dir / 'images',
        csv_dir / 'data' / 'images',
        csv_dir.parent / 'images',
        csv_dir.parent / 'data' / 'images',
        base_path / 'images',
        base_path / 'data' / 'images',
    ])
    
    # Remove duplicates and non-existent dirs
    search_dirs = [d for d in search_dirs if d.exists()]
    
    print(f"Search directories:")
    for d in search_dirs[:5]:  # Show first 5
        print(f"  - {d}")
    
    # Fix paths
    print("\nFixing paths...")
    fixed = 0
    not_found = []
    
    for idx, row in df.iterrows():
        old_path = Path(row['image_path'])
        
        # If already exists and is relative, keep it
        if old_path.exists() and not old_path.is_absolute():
            continue
        
        # Extract filename
        filename = old_path.name
        
        # Try to find image
        new_path = find_image_by_filename(filename, search_dirs)
        
        if new_path:
            # Convert to relative path from base_path
            try:
                rel_path = new_path.relative_to(base_path)
                df.at[idx, 'image_path'] = str(rel_path)
                fixed += 1
            except ValueError:
                # If can't make relative, use absolute
                df.at[idx, 'image_path'] = str(new_path)
                fixed += 1
        else:
            not_found.append(filename)
    
    print(f"Fixed {fixed} paths")
    if not_found:
        print(f"WARNING: {len(not_found)} images not found (first 5: {not_found[:5]})")
    
    # Save
    if output_csv is None:
        output_csv = str(Path(input_csv).with_suffix('.colab.csv'))
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved fixed CSV to: {output_csv}")
    
    # Verify
    print("\nVerifying first 10 paths...")
    verified = 0
    for idx, row in df.head(10).iterrows():
        path = base_path / row['image_path']
        if path.exists():
            verified += 1
    
    print(f"Verified: {verified}/10 paths exist")
    
    return True


def main():
    ap = argparse.ArgumentParser(description="Fix CSV paths for Colab")
    ap.add_argument("input_csv", help="Input CSV file")
    ap.add_argument("--output_csv", default=None, help="Output CSV (default: input.colab.csv)")
    ap.add_argument("--images_dir", default=None, help="Directory containing images")
    ap.add_argument("--base_path", default=None, help="Base path for relative paths")
    
    args = ap.parse_args()
    
    fix_csv_paths(args.input_csv, args.output_csv, args.images_dir, args.base_path)


if __name__ == "__main__":
    main()
