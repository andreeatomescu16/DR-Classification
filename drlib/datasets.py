from pathlib import Path
import cv2, pandas as pd, numpy as np, torch
from torch.utils.data import Dataset

class DRDataset(Dataset):
    def __init__(self, csv_path, split="train", tfm=None, data_root=None):
        """
        Args:
            csv_path: Path to fold CSV
            split: "train" or "val"
            tfm: Albumentations transform
            data_root: Override data root (e.g. /tmp/localdata for local NVMe).
                       Images are read from data_root/raw/.../augmented_resized_V2/...
        """
        self.csv_path = csv_path
        self.data_root = Path(data_root) if data_root else None
        df = pd.read_csv(csv_path)
        # use only rows with label in {0..4}, valid images, and our split
        df = df[(df["label"].isin([0,1,2,3,4])) & (df["is_valid"] == True)]
        if "split" in df.columns:
            df = df[df["split"] == split]
        self.df = df.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self): return len(self.df)

    def _resolve_path(self, row):
        """Resolve image path, optionally using data_root for local storage."""
        img_path = Path(row.image_path)
        if img_path.is_absolute():
            return img_path
        csv_path = Path(self.csv_path)
        folds_dir = csv_path.parent
        data_dir = folds_dir.parent
        full_resolved = (folds_dir / img_path).resolve()
        if self.data_root is not None:
            try:
                path_under_data = full_resolved.relative_to(data_dir)
            except ValueError:
                path_under_data = full_resolved
            return self.data_root / path_under_data
        return full_resolved

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self._resolve_path(row)

        if not img_path.exists():
            img_path = Path(row.image_path)
        if img_path.is_absolute():
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {row.image_path}\n"
                    f"Please check if the file exists or update the CSV with correct paths."
                )
        elif not img_path.exists():
            # Relative path - try multiple resolution strategies
            cwd_path = Path.cwd() / img_path
            if cwd_path.exists():
                img_path = cwd_path
            else:
                csv_path = Path(self.csv_path if hasattr(self, 'csv_path') else '.')
                csv_dir = csv_path.parent if csv_path.is_file() else csv_path
                filename = img_path.name
                possible_paths = [
                    csv_dir / img_path,
                    csv_dir.parent / img_path,
                    Path.cwd() / img_path,
                    Path(img_path),
                ]
                if str(img_path).startswith('DR-Classification/'):
                    possible_paths.insert(0, Path.cwd() / img_path)
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
                if not img_path.exists():
                    for possible_path in possible_paths:
                        if possible_path.exists():
                            img_path = possible_path
                            break
                if not img_path.exists():
                    raise FileNotFoundError(
                        f"Image not found: {row.image_path}\n"
                        f"Tried: {img_path}\n"
                        f"Current directory: {Path.cwd()}\n"
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
            # This should not happen in normal training, but handle gracefully
            img = cv2.resize(img, (512, 512))
            # Convert HWC to CHW format
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            img = torch.from_numpy(img).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
        
        label = int(row.label)
        return img, label
