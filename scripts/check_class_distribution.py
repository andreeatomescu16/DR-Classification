#!/usr/bin/env python3
"""Check class distribution across all folds."""

import pandas as pd
from pathlib import Path
from collections import Counter

def check_fold_distribution(fold_num):
    """Check class distribution for a fold."""
    csv_path = f'data/folds/fold{fold_num}.csv'
    
    if not Path(csv_path).exists():
        return None
    
    df = pd.read_csv(csv_path)
    
    if 'label' not in df.columns:
        print(f"⚠ Fold{fold_num}: No 'label' column found")
        return None
    
    label_counts = Counter(df['label'])
    total = len(df)
    
    return {
        'fold': fold_num,
        'total': total,
        'distribution': dict(sorted(label_counts.items())),
        'percentages': {k: (v/total)*100 for k, v in sorted(label_counts.items())}
    }

def main():
    print("="*70)
    print("VERIFICARE DISTRIBUȚIE CLASE - TOATE FOLD-URILE")
    print("="*70)
    
    all_distributions = {}
    
    for fold_num in range(5):
        dist = check_fold_distribution(fold_num)
        if dist:
            all_distributions[fold_num] = dist
    
    # Display results
    print("\n" + "="*70)
    print("DISTRIBUȚIE COMPLETĂ")
    print("="*70)
    
    for fold_num in sorted(all_distributions.keys()):
        dist = all_distributions[fold_num]
        print(f"\nFold{dist['fold']}:")
        print(f"  Total: {dist['total']:,} samples")
        print(f"  Distribution:")
        for cls in sorted(dist['distribution'].keys()):
            count = dist['distribution'][cls]
            pct = dist['percentages'][cls]
            print(f"    Class {cls}: {count:6,} ({pct:5.2f}%)")
    
    # Check if distributions are identical
    print("\n" + "="*70)
    print("VERIFICARE IDENTITATE DISTRIBUȚII")
    print("="*70)
    
    if len(all_distributions) > 1:
        # Compare distributions
        fold0_dist = all_distributions[0]['distribution']
        
        identical_folds = []
        different_folds = []
        
        for fold_num in sorted(all_distributions.keys()):
            if fold_num == 0:
                continue
            
            dist = all_distributions[fold_num]['distribution']
            
            # Check if distributions are identical
            if dist == fold0_dist:
                identical_folds.append(fold_num)
            else:
                different_folds.append(fold_num)
        
        if identical_folds:
            print(f"\n⚠ WARNING: Fold-urile {identical_folds} au distribuție IDENTICĂ cu Fold0!")
            print(f"   Aceasta indică o problemă în crearea fold-urilor.")
            print(f"   Fold-urile ar trebui să aibă distribuții similare dar NU identice.")
        else:
            print(f"\n✅ Toate fold-urile au distribuții diferite (corect!)")
        
        if different_folds:
            print(f"\n✓ Fold-urile {different_folds} au distribuții diferite (OK)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("STATISTICI SUMMARY")
    print("="*70)
    
    total_samples = sum(d['total'] for d in all_distributions.values())
    print(f"\nTotal samples across all folds: {total_samples:,}")
    
    # Average distribution
    if all_distributions:
        avg_dist = {}
        for cls in range(5):
            counts = [d['distribution'].get(cls, 0) for d in all_distributions.values()]
            avg_dist[cls] = sum(counts) / len(counts) if counts else 0
        
        print(f"\nAverage distribution (across folds):")
        for cls in sorted(avg_dist.keys()):
            avg_count = avg_dist[cls]
            avg_pct = (avg_count / (total_samples / len(all_distributions))) * 100 if total_samples > 0 else 0
            print(f"  Class {cls}: ~{avg_count:,.0f} samples ({avg_pct:.2f}%)")
    
    print("\n" + "="*70)
    print("✅ Verification complete!")
    print("="*70)

if __name__ == "__main__":
    main()
