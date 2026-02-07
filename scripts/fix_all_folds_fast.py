#!/usr/bin/env python3
"""Fix all folds quickly using file index."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

def build_file_index(base_path):
    """Build index of all image files."""
    print("📚 Building file index...")
    base = Path(base_path)
    file_index = {}
    
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
        for img_file in base.rglob(ext):
            filename = img_file.name
            name_no_ext = Path(filename).stem
            
            # Store both full filename and name without extension
            if filename not in file_index:
                file_index[filename] = img_file
            if name_no_ext not in file_index:
                file_index[name_no_ext] = img_file
    
    print(f"✓ Indexed {len(file_index):,} files")
    return file_index

def fix_fold(fold_num, file_index, base_path):
    """Fix a single fold using file index."""
    csv_path = f'data/folds/fold{fold_num}.csv'
    
    if not Path(csv_path).exists():
        print(f"⚠ Fold{fold_num}: CSV not found")
        return None
    
    print(f"\n{'='*70}")
    print(f"Fixing fold{fold_num}...")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    base = Path(base_path)
    
    print(f"Original rows: {len(df):,}")
    
    fixed_count = 0
    already_ok = 0
    missing_count = 0
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Fold{fold_num}"):
        path_str = row['image_path']
        path = Path(path_str)
        
        # Dacă e path absolut de pe laptop, caută în index
        if path.is_absolute() and '/Users/Andreea/' in str(path):
            filename = path.name
            name_no_ext = Path(filename).stem
            
            # Caută în index
            found_path = None
            if filename in file_index:
                found_path = file_index[filename]
            elif name_no_ext in file_index:
                found_path = file_index[name_no_ext]
            
            if found_path and found_path.exists():
                # Găsit! Actualizează path-ul
                new_path = found_path.relative_to(base)
                row['image_path'] = str(new_path)
                valid_rows.append(row)
                fixed_count += 1
            else:
                # Nu există - elimină rândul
                missing_count += 1
        else:
            # Path relativ - verifică dacă există
            full_path = base / path if not path.is_absolute() else path
            if full_path.exists():
                valid_rows.append(row)
                already_ok += 1
            else:
                # Path relativ dar nu există - încercă să-l găsești în index
                filename = path.name
                name_no_ext = Path(filename).stem
                
                found_path = None
                if filename in file_index:
                    found_path = file_index[filename]
                elif name_no_ext in file_index:
                    found_path = file_index[name_no_ext]
                
                if found_path and found_path.exists():
                    new_path = found_path.relative_to(base)
                    row['image_path'] = str(new_path)
                    valid_rows.append(row)
                    fixed_count += 1
                else:
                    missing_count += 1
    
    # Salvează CSV-ul fixat
    df_clean = pd.DataFrame(valid_rows)
    df_clean.to_csv(csv_path, index=False)
    
    print(f"\nResults:")
    print(f"  ✓ Fixed paths: {fixed_count:,}")
    print(f"  ✓ Already OK: {already_ok:,}")
    print(f"  ✗ Missing (removed): {missing_count:,}")
    print(f"  ✅ Valid rows: {len(df_clean):,}")
    print(f"  ✅ Saved: {csv_path}")
    
    return len(df_clean), missing_count

def main():
    base_path = 'DR-Classification'
    
    print("="*70)
    print("FIX TOATE FOLD-URILE (RAPID)")
    print("="*70)
    
    # Construiește index o singură dată
    file_index = build_file_index(base_path)
    
    # Fix toate fold-urile
    results = {}
    for fold_num in [1, 2, 3, 4]:  # Fold0 e deja OK
        result = fix_fold(fold_num, file_index, base_path)
        if result:
            results[fold_num] = result
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    total_valid = sum(r[0] for r in results.values())
    total_missing = sum(r[1] for r in results.values())
    
    print(f"\nTotal:")
    print(f"  ✅ Valid rows: {total_valid:,}")
    print(f"  ✗ Removed (missing): {total_missing:,}")
    
    print(f"\n{'='*70}")
    print("✅ All folds fixed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
