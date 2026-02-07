#!/usr/bin/env python3
"""
Fix CSV paths for Lambda Labs - converts absolute paths to relative paths.
Searches for images in the combined dataset directory.
"""

import pandas as pd
import argparse
from pathlib import Path
import os


def find_image_by_filename(filename, search_dirs):
    """Find image file by filename in search directories."""
    # First try exact filename match
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Try direct match in this directory
        direct_path = search_dir / filename
        if direct_path.exists() and direct_path.is_file():
            return direct_path
    
    # Then try recursive search (slower but more thorough)
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Try recursive search
        try:
            matches = list(search_dir.rglob(filename))
            if matches:
                # Return first match that is actually a file
                for match in matches:
                    if match.is_file():
                        return match
        except (PermissionError, OSError):
            # Skip directories we can't access
            continue
    
    return None


def fix_csv_paths(input_csv, output_csv=None, dataset_dir=None, base_path=None):
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
    
    # Check if we're in a nested DR-Classification directory
    possible_base_paths = [
        base_path,
        base_path / 'DR-Classification',
        base_path.parent / 'DR-Classification' if base_path.name == 'DR-Classification' else base_path,
    ]
    
    # Find the actual base path that contains images
    actual_base = None
    for possible_base in possible_base_paths:
        if possible_base.exists():
            # Check if it contains data/combined_dataset
            combined_dataset = possible_base / 'data' / 'combined_dataset'
            if combined_dataset.exists():
                actual_base = possible_base
                break
    
    if actual_base is None:
        # Fallback: use base_path and search recursively
        actual_base = base_path
    
    print(f"Using base path: {actual_base.absolute()}")
    
    # Add the combined_dataset directory
    combined_dataset = actual_base / 'data' / 'combined_dataset'
    if combined_dataset.exists():
        search_dirs.append(combined_dataset)
        # Add all subdirectories recursively
        try:
            for subdir in combined_dataset.rglob("*"):
                if subdir.is_dir():
                    # Check if it contains image files
                    png_count = len(list(subdir.glob("*.png")))
                    jpg_count = len(list(subdir.glob("*.jpg")))
                    jpeg_count = len(list(subdir.glob("*.jpeg")))
                    if png_count > 0 or jpg_count > 0 or jpeg_count > 0:
                        search_dirs.append(subdir)
        except (PermissionError, OSError) as e:
            print(f"Warning: Could not search all subdirectories: {e}")
    
    # Also search in dataset_dir if provided
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        if not dataset_path.is_absolute():
            dataset_path = actual_base / dataset_path
        if dataset_path.exists():
            search_dirs.append(dataset_path)
            try:
                for subdir in dataset_path.rglob("*"):
                    if subdir.is_dir():
                        if any(subdir.glob("*.png")) or any(subdir.glob("*.jpg")) or any(subdir.glob("*.jpeg")):
                            search_dirs.append(subdir)
            except (PermissionError, OSError):
                pass
    
    # Add common locations
    csv_path = Path(input_csv)
    csv_dir = csv_path.parent if csv_path.is_file() else csv_path
    common_dirs = [
        actual_base / 'data' / 'combined_dataset',
        actual_base / 'DR-Classification' / 'data' / 'combined_dataset',
        csv_dir.parent / 'combined_dataset',
        csv_dir.parent.parent / 'combined_dataset',
    ]
    
    for common_dir in common_dirs:
        if common_dir.exists() and common_dir not in search_dirs:
            search_dirs.append(common_dir)
            try:
                for subdir in common_dir.rglob("*"):
                    if subdir.is_dir():
                        if any(subdir.glob("*.png")) or any(subdir.glob("*.jpg")) or any(subdir.glob("*.jpeg")):
                            if subdir not in search_dirs:
                                search_dirs.append(subdir)
            except (PermissionError, OSError):
                pass
    
    # Remove duplicates while preserving order
    seen = set()
    search_dirs = [d for d in search_dirs if d not in seen and not seen.add(d)]
    
    print(f"\nSearch directories ({len(search_dirs)} total):")
    for d in search_dirs[:15]:  # Show first 15
        img_count = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg"))) + len(list(d.glob("*.jpeg")))
        print(f"  - {d} ({img_count} images)")
    if len(search_dirs) > 15:
        print(f"  ... and {len(search_dirs) - 15} more")
    
    # Fix paths
    print("\nFixing paths...")
    fixed = 0
    not_found = []
    
    for idx, row in df.iterrows():
        old_path = Path(row['image_path'])
        
        # If already exists and is relative, keep it
        if old_path.exists() and not old_path.is_absolute():
            continue
        
        # If absolute path, try to find image
        if old_path.is_absolute():
            filename = old_path.name
            
            # Try to find image
            new_path = find_image_by_filename(filename, search_dirs)
            
            if new_path:
                # Convert to relative path from actual_base (found above)
                # Find actual_base again (same logic as above)
                possible_base_paths = [
                    base_path,
                    base_path / 'DR-Classification',
                    base_path.parent / 'DR-Classification' if base_path.name == 'DR-Classification' else base_path,
                ]
                actual_base = base_path
                for possible_base in possible_base_paths:
                    if possible_base.exists():
                        combined_dataset = possible_base / 'data' / 'combined_dataset'
                        if combined_dataset.exists():
                            actual_base = possible_base
                            break
                
                try:
                    rel_path = new_path.relative_to(actual_base)
                    df.at[idx, 'image_path'] = str(rel_path)
                    fixed += 1
                except ValueError:
                    # If can't make relative, try relative to current working directory
                    try:
                        rel_path = new_path.relative_to(Path.cwd())
                        df.at[idx, 'image_path'] = str(rel_path)
                        fixed += 1
                    except ValueError:
                        # Last resort: use absolute path
                        df.at[idx, 'image_path'] = str(new_path)
                        fixed += 1
            else:
                not_found.append(filename)
    
    print(f"Fixed {fixed} paths")
    if not_found:
        print(f"WARNING: {len(not_found)} images not found (first 10: {not_found[:10]})")
        if len(not_found) > len(df) * 0.1:  # More than 10% missing
            print("⚠ ERROR: Too many images missing! Check dataset directory.")
            return False
    
    # Save
    if output_csv is None:
        output_csv = str(Path(input_csv).with_suffix('.fixed.csv'))
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved fixed CSV to: {output_csv}")
    
    # Verify
    print("\nVerifying first 10 paths...")
    verified = 0
    # Find actual_base again for verification
    possible_base_paths = [
        base_path,
        base_path / 'DR-Classification',
        base_path.parent / 'DR-Classification' if base_path.name == 'DR-Classification' else base_path,
    ]
    actual_base = base_path
    for possible_base in possible_base_paths:
        if possible_base.exists():
            combined_dataset = possible_base / 'data' / 'combined_dataset'
            if combined_dataset.exists():
                actual_base = possible_base
                break
    
    for idx, row in df.head(10).iterrows():
        path_str = row['image_path']
        # Try relative to actual_base first
        path = actual_base / path_str
        if not path.exists():
            # Try relative to current directory
            path = Path(path_str)
            if not path.is_absolute():
                path = Path.cwd() / path_str
        if path.exists():
            verified += 1
    
    print(f"Verified: {verified}/10 paths exist")
    
    return True


def main():
    ap = argparse.ArgumentParser(description="Fix CSV paths for Lambda Labs")
    ap.add_argument("input_csv", help="Input CSV file (e.g., data/folds/fold0.csv)")
    ap.add_argument("--output_csv", default=None, help="Output CSV (default: input.fixed.csv)")
    ap.add_argument("--dataset_dir", default="data/combined_dataset", help="Directory containing images")
    ap.add_argument("--base_path", default=None, help="Base path for relative paths")
    
    args = ap.parse_args()
    
    success = fix_csv_paths(args.input_csv, args.output_csv, args.dataset_dir, args.base_path)
    
    if success:
        print("\n✅ Path fixing complete!")
        print(f"Use the fixed CSV: {args.output_csv or Path(args.input_csv).with_suffix('.fixed.csv')}")
    else:
        print("\n❌ Path fixing failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
