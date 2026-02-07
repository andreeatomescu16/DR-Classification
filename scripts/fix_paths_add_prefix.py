#!/usr/bin/env python3
"""Add DR-Classification/ prefix to all paths in CSV."""

import pandas as pd
from pathlib import Path

def fix_csv_paths(csv_path, prefix="DR-Classification"):
    """Add prefix to all relative paths in CSV."""
    print(f"Processing: {csv_path}")
    df = pd.read_csv(csv_path)
    
    original_count = len(df)
    fixed_count = 0
    
    for idx, row in df.iterrows():
        path_str = row['image_path']
        path = Path(path_str)
        
        # Dacă e path relativ și nu începe cu prefix-ul
        if not path.is_absolute() and not str(path).startswith(prefix):
            # Adaugă prefix
            new_path = f"{prefix}/{path_str}"
            df.at[idx, 'image_path'] = new_path
            fixed_count += 1
    
    # Salvează CSV-ul fixat
    df.to_csv(csv_path, index=False)
    
    print(f"  Fixed {fixed_count}/{original_count} paths")
    print(f"  ✅ Saved: {csv_path}")

def main():
    print("="*70)
    print("FIX PATH-URI - ADAUGARE PREFIX DR-Classification/")
    print("="*70)
    
    # Fix Fold0
    fix_csv_paths('data/folds/fold0.csv')
    
    print("\n" + "="*70)
    print("✅ Path fixing complete!")
    print("="*70)
    
    # Verificare rapidă
    print("\nVerifying first path...")
    df = pd.read_csv('data/folds/fold0.csv')
    first_path = df.iloc[0]['image_path']
    print(f"First path: {first_path}")
    
    full_path = Path('DR-Classification') / first_path if not Path(first_path).is_absolute() else Path(first_path)
    print(f"Full path: {full_path}")
    print(f"Exists: {full_path.exists()}")

if __name__ == "__main__":
    main()
