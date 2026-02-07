#!/usr/bin/env python3
"""Check dataset structure to understand where images are located."""

import argparse
from pathlib import Path


def check_structure(dataset_dir):
    """Check the structure of the dataset directory."""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"ERROR: Directory does not exist: {dataset_path}")
        return
    
    print(f"Checking structure of: {dataset_path}")
    print("=" * 60)
    
    # Count images
    png_files = list(dataset_path.rglob("*.png"))
    jpg_files = list(dataset_path.rglob("*.jpg"))
    jpeg_files = list(dataset_path.rglob("*.jpeg"))
    
    print(f"\nImage files found:")
    print(f"  PNG: {len(png_files)}")
    print(f"  JPG: {len(jpg_files)}")
    print(f"  JPEG: {len(jpeg_files)}")
    print(f"  Total: {len(png_files) + len(jpg_files) + len(jpeg_files)}")
    
    # Show directory structure (first 3 levels)
    print(f"\nDirectory structure (first 3 levels):")
    print_structure(dataset_path, max_depth=3, current_depth=0)
    
    # Find directories with images
    print(f"\nDirectories containing images:")
    image_dirs = set()
    for img_file in (png_files + jpg_files + jpeg_files)[:100]:  # Sample first 100
        image_dirs.add(img_file.parent)
    
    for img_dir in sorted(list(image_dirs))[:20]:  # Show first 20
        img_count = len(list(img_dir.glob("*.png"))) + len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.jpeg")))
        print(f"  {img_dir.relative_to(dataset_path)} ({img_count} images)")
    
    # Sample filenames
    print(f"\nSample filenames:")
    all_files = (png_files + jpg_files + jpeg_files)[:10]
    for f in all_files:
        print(f"  {f.name} -> {f.relative_to(dataset_path)}")


def print_structure(path, max_depth=3, current_depth=0, prefix=""):
    """Print directory structure."""
    if current_depth >= max_depth:
        return
    
    if path.is_dir():
        try:
            items = sorted([p for p in path.iterdir() if p.is_dir()])[:10]  # First 10 dirs
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}/")
                
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_structure(item, max_depth, current_depth + 1, next_prefix)
        except PermissionError:
            print(f"{prefix}└── [Permission denied]")


def main():
    ap = argparse.ArgumentParser(description="Check dataset structure")
    ap.add_argument("--dataset_dir", default="data/combined_dataset", help="Dataset directory")
    
    args = ap.parse_args()
    check_structure(args.dataset_dir)


if __name__ == "__main__":
    main()
