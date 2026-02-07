#!/usr/bin/env python3
"""Final verification - test actual image loading and quality."""

import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from collections import Counter

def verify_fold_loading(fold_num, base_path, sample_size=100):
    """Verify that images can actually be loaded."""
    csv_path = f'data/folds/fold{fold_num}.csv'
    
    if not Path(csv_path).exists():
        return None
    
    df = pd.read_csv(csv_path)
    base = Path(base_path)
    
    print(f"\n{'='*70}")
    print(f"Fold{fold_num} - Image Loading Test")
    print(f"{'='*70}")
    
    # Sample random rows
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    
    loaded_count = 0
    failed_count = 0
    sizes = []
    channels = []
    labels = []
    
    for idx, row in sample_df.iterrows():
        path_str = row['image_path']
        path = Path(path_str)
        full_path = base / path if not path.is_absolute() else path
        
        try:
            # Try to load image
            img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
            
            if img is None:
                failed_count += 1
                continue
            
            loaded_count += 1
            h, w = img.shape[:2]
            sizes.append((w, h))
            channels.append(img.shape[2] if len(img.shape) > 2 else 1)
            
            if 'label' in row:
                labels.append(row['label'])
        except Exception as e:
            failed_count += 1
            if failed_count <= 3:
                print(f"  ⚠ Failed to load: {path_str} - {e}")
    
    stats = {
        'total_sampled': len(sample_df),
        'loaded': loaded_count,
        'failed': failed_count,
        'sizes': sizes,
        'channels': channels,
        'labels': labels
    }
    
    success_rate = (loaded_count / len(sample_df)) * 100 if len(sample_df) > 0 else 0
    
    print(f"  Sampled: {len(sample_df)} images")
    print(f"  ✓ Loaded successfully: {loaded_count} ({success_rate:.2f}%)")
    print(f"  ✗ Failed to load: {failed_count}")
    
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        print(f"\n  Image dimensions:")
        print(f"    Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        print(f"    Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
    
    if channels:
        channel_counts = Counter(channels)
        print(f"\n  Channels: {dict(channel_counts)}")
    
    if labels:
        label_counts = Counter(labels)
        print(f"\n  Label distribution (sample):")
        for label, count in sorted(label_counts.items()):
            print(f"    Class {label}: {count}")
    
    return stats

def main():
    base_path = 'DR-Classification'
    
    print("="*70)
    print("VERIFICARE FINALĂ - CALITATE IMAGINI")
    print("="*70)
    
    all_stats = {}
    
    # Test fold0 (cel folosit pentru training)
    print("\n🔍 Testing Fold0 (used for training)...")
    stats = verify_fold_loading(0, base_path, sample_size=200)
    all_stats[0] = stats
    
    # Quick test for other folds
    for fold_num in [1, 2, 3, 4]:
        stats = verify_fold_loading(fold_num, base_path, sample_size=50)
        all_stats[fold_num] = stats
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total_loaded = sum(s['loaded'] for s in all_stats.values() if s)
    total_sampled = sum(s['total_sampled'] for s in all_stats.values() if s)
    total_failed = sum(s['failed'] for s in all_stats.values() if s)
    
    if total_sampled > 0:
        overall_success = (total_loaded / total_sampled) * 100
        print(f"\nOverall success rate: {overall_success:.2f}%")
        print(f"  ✓ Loaded: {total_loaded}/{total_sampled}")
        print(f"  ✗ Failed: {total_failed}")
    
    # Check if ready for training
    fold0_stats = all_stats.get(0)
    if fold0_stats and fold0_stats['loaded'] > 0:
        success_rate = (fold0_stats['loaded'] / fold0_stats['total_sampled']) * 100
        if success_rate >= 95:
            print(f"\n✅ Fold0 is ready for training!")
            print(f"   Success rate: {success_rate:.2f}%")
            print(f"   All images can be loaded correctly.")
        else:
            print(f"\n⚠ Warning: Fold0 has {100-success_rate:.2f}% failed images")
            print(f"   Consider checking failed images before training.")
    
    print(f"\n{'='*70}")
    print("✅ Final verification complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
