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
        img = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tfm:
            aug = self.tfm(image=img)
            img = aug["image"]
        img = img.float() / 255.0
        # normalize to ImageNet stats
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img = (img - mean) / std
        label = int(row.label)
        return img, label
