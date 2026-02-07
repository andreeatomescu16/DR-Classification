#!/usr/bin/env python3
"""Find where images are actually located."""

from pathlib import Path
import sys

def find_images(base_path="."):
    """Find all directories containing images."""
    base = Path(base_path)
    
    print(f"Searching for images in: {base.absolute()}")
    print("=" * 60)
    
    image_dirs = {}
    
    # Search recursively
    for img_file in base.rglob("*.png"):
        img_dir = img_file.parent
        if img_dir not in image_dirs:
            image_dirs[img_dir] = {"png": 0, "jpg": 0, "jpeg": 0}
        image_dirs[img_dir]["png"] += 1
    
    for img_file in base.rglob("*.jpg"):
        img_dir = img_file.parent
        if img_dir not in image_dirs:
            image_dirs[img_dir] = {"png": 0, "jpg": 0, "jpeg": 0}
        image_dirs[img_dir]["jpg"] += 1
    
    for img_file in base.rglob("*.jpeg"):
        img_dir = img_file.parent
        if img_dir not in image_dirs:
            image_dirs[img_dir] = {"png": 0, "jpg": 0, "jpeg": 0}
        image_dirs[img_dir]["jpeg"] += 1
    
    # Sort by total image count
    sorted_dirs = sorted(image_dirs.items(), 
                        key=lambda x: sum(x[1].values()), 
                        reverse=True)
    
    print(f"\nFound {len(sorted_dirs)} directories with images")
    print(f"\nTop 20 directories:")
    for i, (img_dir, counts) in enumerate(sorted_dirs[:20], 1):
        total = sum(counts.values())
        rel_path = img_dir.relative_to(base)
        print(f"{i:2d}. {rel_path}")
        print(f"    PNG: {counts['png']}, JPG: {counts['jpg']}, JPEG: {counts['jpeg']}, Total: {total}")
    
    # Sample filenames from top directories
    print(f"\nSample filenames from top directories:")
    for img_dir, counts in sorted_dirs[:3]:
        rel_path = img_dir.relative_to(base)
        sample_files = list(img_dir.glob("*.png"))[:3] + list(img_dir.glob("*.jpg"))[:3]
        print(f"\n{rel_path}:")
        for f in sample_files[:5]:
            print(f"  - {f.name}")
    
    return sorted_dirs

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    find_images(base)
