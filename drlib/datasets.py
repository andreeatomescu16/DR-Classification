from pathlib import Path
import cv2, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, csv_path, split="train", tfm=None):
        self.csv_path = csv_path
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
        
        # Try to load image, with fallback path resolution
        img_path = Path(row.image_path)
        
        # If path is absolute, use it directly
        if img_path.is_absolute():
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {row.image_path}\n"
                    f"Please check if the file exists or update the CSV with correct paths."
                )
        else:
            # Relative path - try multiple resolution strategies
            if not img_path.exists():
                # Strategy 1: Try relative to current working directory
                cwd_path = Path.cwd() / img_path
                if cwd_path.exists():
                    img_path = cwd_path
                else:
                    # Strategy 2: Try relative to CSV file location
                    csv_path = Path(self.csv_path if hasattr(self, 'csv_path') else '.')
                    csv_dir = csv_path.parent if csv_path.is_file() else csv_path
                    
                    # Extract filename
                    filename = img_path.name
                    
                    # Try common locations relative to CSV
                    possible_paths = [
                        csv_dir / img_path,  # Same relative path from CSV dir
                        csv_dir.parent / img_path,  # One level up
                        Path.cwd() / img_path,  # From current directory
                        Path(img_path),  # Direct (in case we're already in right dir)
                    ]
                    
                    # Strategy 3: If path starts with DR-Classification/, try from current dir
                    if str(img_path).startswith('DR-Classification/'):
                        possible_paths.insert(0, Path.cwd() / img_path)
                    
                    # Strategy 4: Search recursively for filename
                    if csv_dir.exists():
                        for possible_dir in [csv_dir, csv_dir.parent, Path.cwd()]:
                            if possible_dir.exists():
                                try:
                                    matches = list(possible_dir.rglob(filename))
                                    if matches:
                                        img_path = matches[0]
                                        break
                                except (PermissionError, OSError):
                                    continue
                    
                    # Try the possible_paths
                    if not img_path.exists():
                        for possible_path in possible_paths:
                            if possible_path.exists():
                                img_path = possible_path
                                break
                    
                    # If still not found, raise error
                    if not img_path.exists():
                        raise FileNotFoundError(
                            f"Image not found: {row.image_path}\n"
                            f"Tried: {img_path}\n"
                            f"Current directory: {Path.cwd()}\n"
                            f"Please check if the file exists or update the CSV with correct paths.\n"
                            f"For Colab: Make sure images are uploaded to Google Drive and paths are relative."
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
