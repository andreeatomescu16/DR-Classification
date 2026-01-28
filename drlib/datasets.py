from pathlib import Path
import cv2, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, csv_path, split="train", tfm=None):
        df = pd.read_csv(csv_path)
        # use only rows with label in {0..4}, valid images, and our split
        df = df[(df["label"].isin([0,1,2,3,4])) & (df["is_valid"] == True)]
        if "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        
        # Try to load image, skip if not found
        img_path = Path(row.image_path)
        if not img_path.exists():
            # Try to find alternative paths or use a placeholder
            # For now, we'll raise an error but with better message
            raise FileNotFoundError(
                f"Image not found: {row.image_path}\n"
                f"Please check if the file exists or update the CSV with correct paths."
            )
        
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not load image (may be corrupted): {row.image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (includes normalization and tensor conversion)
        if self.tfm:
            aug = self.tfm(image=img)
            img = aug["image"]
        else:
            # Fallback: basic preprocessing if no transform provided
            img = cv2.resize(img, (512, 512))
            img = torch.from_numpy(img).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
        
        label = int(row.label)
        return img, label
