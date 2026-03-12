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
        """
        Resolve image path, optionally using data_root for local storage.

        When data_root is provided (e.g. /tmp/localdata on RunPod), we want a
        fast, deterministic mapping from the CSV's image_path to the local tree.

        The fold CSVs store image_path relative to the folds dir, typically like:
            ../raw/augmented_resized_V2/...

        For data_root=/tmp/localdata this should become:
            /tmp/localdata/raw/augmented_resized_V2/...

        In this case we avoid any .resolve()/relative_to()/rglob logic entirely.
        """
        img_path = Path(row.image_path)

        # Absolute paths are used as-is (debug/manual overrides).
        if img_path.is_absolute():
            return img_path

        # Fast path: map relative CSV path onto data_root.
        # Fold CSV stores paths like: ../raw/augmented_resized_V2/train/3/file.jpg
        # For data_root=/tmp/localdata we want:
        #   /tmp/localdata/augmented_resized_V2/train/3/file.jpg
        if self.data_root is not None:
            parts = list(img_path.parts)
            # Strip leading ".." if present
            if parts and parts[0] == "..":
                parts = parts[1:]
            # Strip leading "raw" if present
            if parts and parts[0] == "raw":
                parts = parts[1:]
            img_rel = Path(*parts) if parts else Path()
            return self.data_root / img_rel

        # Fallback path (no data_root): resolve relative to folds directory.
        csv_path = Path(self.csv_path)
        folds_dir = csv_path.parent
        return (folds_dir / img_path).resolve()

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self._resolve_path(row)

        # Optional debug: print first few resolved paths when enabled.
        # This is gated by DRLIB_DEBUG_PATHS to avoid log spam.
        if i < 5:
            import os
            if os.getenv("DRLIB_DEBUG_PATHS") == "1":
                print(
                    f"[DRLIB_DEBUG_PATHS] idx={i} "
                    f"csv_path={row.image_path} "
                    f"resolved={img_path} "
                    f"exists={img_path.exists()} "
                    f"data_root={self.data_root}"
                )

        # When data_root is set, we *do not* attempt any expensive filesystem
        # scans (no rglob). If the mapped path does not exist this is an error.
        if self.data_root is not None:
            if not img_path.exists():
                raise FileNotFoundError(
                    "Image not found under data_root.\n"
                    f"  original image_path: {row.image_path}\n"
                    f"  resolved path      : {img_path}\n"
                    f"  data_root          : {self.data_root}\n"
                    "Please check that DATA_ROOT matches the dataset layout or "
                    "rebuild the folds / copy the data correctly."
                )
        else:
            # Legacy fallback path resolution (no data_root). This keeps the
            # previous behaviour for local experiments but may be expensive on
            # very large trees, so it's intentionally *not* used with data_root.
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
