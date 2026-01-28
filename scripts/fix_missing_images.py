#!/usr/bin/env python3
"""
Script to identify and optionally remove rows with missing images from CSV files.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def check_images(csv_path, output_path=None, remove_missing=True):
    """Check for missing images and optionally create cleaned CSV."""
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Total rows: {len(df)}")
    
    # Check which images exist
    print("\nChecking image paths...")
    missing_paths = []
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(row['image_path'])
        if img_path.exists():
            valid_indices.append(idx)
        else:
            missing_paths.append(row['image_path'])
    
    print(f"\nValid images: {len(valid_indices)}")
    print(f"Missing images: {len(missing_paths)}")
    
    if missing_paths:
        print("\nFirst 10 missing paths:")
        for path in missing_paths[:10]:
            print(f"  - {path}")
    
    if remove_missing and output_path:
        # Create cleaned dataframe
        df_clean = df.loc[valid_indices].reset_index(drop=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\nSaved cleaned CSV to: {output_path}")
        print(f"Removed {len(df) - len(df_clean)} rows with missing images")
    
    return missing_paths, valid_indices


def main():
    ap = argparse.ArgumentParser(description="Check and fix missing images in CSV")
    ap.add_argument("csv_path", help="Input CSV file")
    ap.add_argument("--output", help="Output CSV file (if not provided, only reports)")
    ap.add_argument("--remove", action='store_true', help="Remove rows with missing images")
    
    args = ap.parse_args()
    
    output_path = args.output if args.output else None
    if args.remove and not output_path:
        output_path = str(Path(args.csv_path).with_suffix('.cleaned.csv'))
        print(f"Output file not specified, using: {output_path}")
    
    missing, valid = check_images(args.csv_path, output_path, args.remove)


if __name__ == "__main__":
    main()

