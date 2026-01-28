#!/usr/bin/env python3
"""
Dataset analysis and visualization script.

This script analyzes dataset statistics including:
- Class distribution
- Image size distribution
- Dataset source breakdown
- Sample visualizations
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_dataset(csv_path, output_dir=None):
    """Analyze dataset and generate visualizations."""
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Filter valid images
    df_valid = df[df['is_valid'] == True].copy()
    df_labeled = df_valid[df_valid['label'].isin([0, 1, 2, 3, 4])].copy()
    
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    print(f"Total rows: {len(df)}")
    print(f"Valid images: {len(df_valid)}")
    print(f"Labeled images: {len(df_labeled)}")
    print(f"Invalid images: {len(df) - len(df_valid)}")
    print(f"Unlabeled images: {len(df_valid) - len(df_labeled)}")
    
    # Class distribution
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    class_counts = df_labeled['label'].value_counts().sort_index()
    class_names = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']
    
    for i, count in enumerate(class_counts):
        pct = count / len(df_labeled) * 100
        print(f"{class_names[i]}: {count:6d} ({pct:5.2f}%)")
    
    # Dataset source breakdown
    if 'dataset' in df_labeled.columns:
        print("\n" + "="*60)
        print("DATASET SOURCE BREAKDOWN")
        print("="*60)
        source_counts = df_labeled['dataset'].value_counts()
        for source, count in source_counts.items():
            pct = count / len(df_labeled) * 100
            print(f"{source}: {count:6d} ({pct:5.2f}%)")
    
    # Split distribution
    if 'split' in df_labeled.columns:
        print("\n" + "="*60)
        print("SPLIT DISTRIBUTION")
        print("="*60)
        split_counts = df_labeled['split'].value_counts()
        for split, count in split_counts.items():
            pct = count / len(df_labeled) * 100
            print(f"{split}: {count:6d} ({pct:5.2f}%)")
    
    # Image size statistics
    if 'width' in df_labeled.columns and 'height' in df_labeled.columns:
        print("\n" + "="*60)
        print("IMAGE SIZE STATISTICS")
        print("="*60)
        widths = df_labeled['width'].dropna()
        heights = df_labeled['height'].dropna()
        if len(widths) > 0:
            print(f"Width:  min={widths.min()}, max={widths.max()}, mean={widths.mean():.1f}, median={widths.median():.1f}")
            print(f"Height: min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}, median={heights.median():.1f}")
    
    # Generate visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class distribution plot
        plt.figure(figsize=(10, 6))
        class_counts.plot(kind='bar', color='steelblue')
        plt.xlabel('DR Severity Level')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.xticks(range(5), class_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution.png', dpi=150)
        print(f"\nSaved class distribution plot to {output_dir / 'class_distribution.png'}")
        
        # Class distribution pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(class_counts.values, labels=class_names, autopct='%1.1f%%', startangle=90)
        plt.title('Class Distribution (Pie Chart)')
        plt.tight_layout()
        plt.savefig(output_dir / 'class_distribution_pie.png', dpi=150)
        print(f"Saved pie chart to {output_dir / 'class_distribution_pie.png'}")
        
        # Image size distribution
        if len(widths) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            widths_clean = widths.dropna()
            heights_clean = heights.dropna()
            if len(widths_clean) > 0:
                ax1.hist(widths_clean.values, bins=50, color='steelblue', alpha=0.7)
                ax1.set_xlabel('Width (pixels)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Image Width Distribution')
                ax1.grid(True, alpha=0.3)
            
            if len(heights_clean) > 0:
                ax2.hist(heights_clean.values, bins=50, color='steelblue', alpha=0.7)
                ax2.set_xlabel('Height (pixels)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Image Height Distribution')
                ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'image_size_distribution.png', dpi=150)
            print(f"Saved image size distribution to {output_dir / 'image_size_distribution.png'}")
        
        # Dataset source breakdown
        if 'dataset' in df_labeled.columns:
            plt.figure(figsize=(10, 6))
            source_counts.plot(kind='bar', color='coral')
            plt.xlabel('Dataset Source')
            plt.ylabel('Count')
            plt.title('Dataset Source Breakdown')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(output_dir / 'dataset_source.png', dpi=150)
            print(f"Saved dataset source breakdown to {output_dir / 'dataset_source.png'}")
        
        # Sample images visualization
        print("\nGenerating sample images visualization...")
        visualize_samples(df_labeled, output_dir / 'sample_images.png', n_samples=20)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


def visualize_samples(df, save_path, n_samples=20):
    """Visualize sample images from each class."""
    n_classes = 5
    samples_per_class = n_samples // n_classes
    
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(15, 15))
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    for class_idx in range(n_classes):
        class_df = df[df['label'] == class_idx]
        if len(class_df) == 0:
            continue
        
        samples = class_df.sample(min(samples_per_class, len(class_df)), random_state=42)
        
        for col_idx, (_, row) in enumerate(samples.iterrows()):
            try:
                img_path = row['image_path']
                img = Image.open(img_path)
                img.thumbnail((224, 224))
                
                axes[class_idx, col_idx].imshow(img)
                axes[class_idx, col_idx].axis('off')
                if col_idx == 0:
                    axes[class_idx, col_idx].set_ylabel(class_names[class_idx], fontsize=12)
            except Exception as e:
                axes[class_idx, col_idx].text(0.5, 0.5, 'Error', ha='center', va='center')
                axes[class_idx, col_idx].axis('off')
    
    plt.suptitle('Sample Images by Class', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved sample images to {save_path}")


def main():
    ap = argparse.ArgumentParser(description="Analyze DR classification dataset")
    ap.add_argument("csv_path", help="Path to dataset CSV file")
    ap.add_argument("--output_dir", default=None, help="Output directory for visualizations")
    args = ap.parse_args()
    
    analyze_dataset(args.csv_path, args.output_dir)


if __name__ == "__main__":
    main()

