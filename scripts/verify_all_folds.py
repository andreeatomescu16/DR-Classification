#!/usr/bin/env python3
"""Verify all folds and report statistics about image paths."""

import pandas as pd
from pathlib import Path
from collections import defaultdict

def verify_fold(fold_num, base_path):
    """Verify a single fold and return statistics."""
    csv_path = f'data/folds/fold{fold_num}.csv'
    
    if not Path(csv_path).exists():
        return None
    
    df = pd.read_csv(csv_path)
    base = Path(base_path)
    
    stats = {
        'total_rows': len(df),
        'exists': 0,
        'missing': 0,
        'absolute_laptop_paths': 0,
        'relative_paths': 0,
        'fixed_absolute': 0,
        'missing_filenames': []
    }
    
    for idx, row in df.iterrows():
        path_str = row['image_path']
        path = Path(path_str)
        
        # Check if it's an absolute path from laptop
        if path.is_absolute() and '/Users/Andreea/' in str(path):
            stats['absolute_laptop_paths'] += 1
            filename = path.name
            
            # Try to find it
            found = False
            name_no_ext = Path(filename).stem
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                matches = list(base.rglob(f'{name_no_ext}{ext}'))
                if matches and matches[0].exists():
                    stats['fixed_absolute'] += 1
                    found = True
                    break
            
            if not found:
                stats['missing'] += 1
                if len(stats['missing_filenames']) < 10:
                    stats['missing_filenames'].append(filename)
        else:
            # Relative path
            stats['relative_paths'] += 1
            full_path = base / path if not path.is_absolute() else path
            
            if full_path.exists():
                stats['exists'] += 1
            else:
                stats['missing'] += 1
                if len(stats['missing_filenames']) < 10:
                    stats['missing_filenames'].append(path.name)
    
    return stats

def main():
    base_path = 'DR-Classification'
    
    print("="*70)
    print("VERIFICARE TOATE FOLD-URILE")
    print("="*70)
    
    all_stats = {}
    
    for fold_num in range(5):
        print(f"\n{'='*70}")
        print(f"Fold{fold_num}:")
        print(f"{'='*70}")
        
        stats = verify_fold(fold_num, base_path)
        
        if stats is None:
            print(f"  ⚠ CSV not found: data/folds/fold{fold_num}.csv")
            continue
        
        all_stats[fold_num] = stats
        
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  ✓ Paths exist: {stats['exists']:,} ({stats['exists']/stats['total_rows']*100:.2f}%)")
        print(f"  ✗ Missing: {stats['missing']:,} ({stats['missing']/stats['total_rows']*100:.2f}%)")
        print(f"  📍 Absolute laptop paths: {stats['absolute_laptop_paths']:,}")
        print(f"  📍 Relative paths: {stats['relative_paths']:,}")
        print(f"  🔧 Fixed absolute paths: {stats['fixed_absolute']:,}")
        
        if stats['missing_filenames']:
            print(f"\n  Missing filenames (first 10):")
            for fn in stats['missing_filenames']:
                print(f"    - {fn}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - TOATE FOLD-URILE")
    print(f"{'='*70}")
    
    total_rows = sum(s['total_rows'] for s in all_stats.values())
    total_exists = sum(s['exists'] for s in all_stats.values())
    total_missing = sum(s['missing'] for s in all_stats.values())
    total_absolute = sum(s['absolute_laptop_paths'] for s in all_stats.values())
    total_fixed = sum(s['fixed_absolute'] for s in all_stats.values())
    
    print(f"\nTotal across all folds:")
    print(f"  Total rows: {total_rows:,}")
    print(f"  ✓ Paths exist: {total_exists:,} ({total_exists/total_rows*100:.2f}%)")
    print(f"  ✗ Missing: {total_missing:,} ({total_missing/total_rows*100:.2f}%)")
    print(f"  📍 Absolute laptop paths: {total_absolute:,}")
    print(f"  🔧 Fixed absolute paths: {total_fixed:,}")
    print(f"  ⚠ Still missing: {total_absolute - total_fixed:,}")
    
    # Per-fold summary
    print(f"\nPer-fold breakdown:")
    print(f"{'Fold':<6} {'Total':<10} {'Exists':<10} {'Missing':<10} {'% OK':<8}")
    print("-" * 50)
    for fold_num in sorted(all_stats.keys()):
        s = all_stats[fold_num]
        pct_ok = s['exists'] / s['total_rows'] * 100
        print(f"{fold_num:<6} {s['total_rows']:<10,} {s['exists']:<10,} {s['missing']:<10,} {pct_ok:<8.2f}%")
    
    print(f"\n{'='*70}")
    print("✅ Verification complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
