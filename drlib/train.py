"""
Training module for Diabetic Retinopathy classification using PyTorch Lightning.

This module provides:
- Lightning module with comprehensive metrics tracking
- Support for different loss functions (CE, weighted CE, focal loss)
- Learning rate scheduling
- Freeze/unfreeze training strategies
- Class weight computation from data
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping

from drlib.datasets import DRDataset
from drlib.transforms import get_train_tf, get_val_tf
from drlib.models import create_model
from drlib.metrics import compute_all_metrics, quadratic_weighted_kappa
from drlib.losses import create_loss


class DRModule(L.LightningModule):
    """
    PyTorch Lightning module for DR classification.
    
    Features:
    - Multiple loss function options
    - Comprehensive metrics tracking
    - Learning rate scheduling
    - Freeze/unfreeze training strategies
    """
    def __init__(
        self,
        model_name="efficientnet_b3",
        lr=1e-4,
        img_size=512,
        loss_type='ce',
        class_counts=None,
        freeze_backbone=False,
        unfreeze_epoch=None,
        lr_scheduler='cosine',
        weight_decay=1e-4,
        warmup_epochs=0,
        **loss_kwargs
    ):
        """
        Args:
            model_name: Model architecture name
            lr: Initial learning rate
            img_size: Image size (square)
            loss_type: 'ce', 'weighted_ce', 'focal', or 'label_smoothing'
            class_counts: Array of class frequencies (for weighted losses)
            freeze_backbone: Whether to freeze backbone initially
            unfreeze_epoch: Epoch to unfreeze backbone (None = never freeze)
            lr_scheduler: 'cosine', 'step', 'plateau', or None
            weight_decay: Weight decay for optimizer
            warmup_epochs: Number of warmup epochs
            **loss_kwargs: Additional arguments for loss function
        """
        super().__init__()
        self.save_hyperparameters(ignore=['class_counts'])
        
        # Create model
        self.model = create_model(model_name, num_classes=5, pretrained=True)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Create loss function (filter kwargs based on loss type)
        # Note: smoothing is used internally for weight computation, not passed to loss
        filtered_kwargs = {}
        if loss_type == 'focal':
            filtered_kwargs['gamma'] = loss_kwargs.get('gamma', 2.0)
        elif loss_type == 'label_smoothing':
            filtered_kwargs['num_classes'] = 5
            filtered_kwargs['smoothing'] = loss_kwargs.get('label_smoothing', 0.1)
        
        self.criterion = create_loss(
            loss_type=loss_type,
            class_counts=class_counts,
            smoothing=loss_kwargs.get('smoothing', 1.0) if loss_type in ['weighted_ce', 'focal'] else None,
            **filtered_kwargs
        )
        
        # Metrics storage
        self.train_preds, self.train_targs, self.train_probs = [], [], []
        self.val_preds, self.val_targs, self.val_probs = [], [], []
        
        self.unfreeze_epoch = unfreeze_epoch
    
    def _freeze_backbone(self):
        """Freeze all layers except classifier."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'head' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all layers."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Store predictions for metrics
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        
        self.train_preds.extend(preds.detach().cpu().numpy().tolist())
        self.train_targs.extend(y.detach().cpu().numpy().tolist())
        self.train_probs.extend(probs.detach().cpu().numpy().tolist())
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def on_train_epoch_start(self):
        """Unfreeze backbone at specified epoch."""
        if self.unfreeze_epoch is not None and self.current_epoch == self.unfreeze_epoch:
            print(f"\nUnfreezing backbone at epoch {self.current_epoch}")
            self._unfreeze_backbone()
    
    def on_train_epoch_end(self):
        """Compute training metrics."""
        if self.train_targs:
            metrics = compute_all_metrics(
                self.train_targs,
                self.train_preds,
                np.array(self.train_probs),
                n_classes=5
            )
            
            # Log key metrics
            self.log("train_qwk", metrics['qwk'], prog_bar=True, on_epoch=True)
            self.log("train_acc", metrics['accuracy'], prog_bar=False, on_epoch=True)
            self.log("train_f1", metrics['macro_f1'], prog_bar=False, on_epoch=True)
            
            # Log per-class metrics
            for i in range(5):
                self.log(f"train_precision_class_{i}", metrics[f'precision_class_{i}'], on_epoch=True)
                self.log(f"train_recall_class_{i}", metrics[f'recall_class_{i}'], on_epoch=True)
        
        # Reset storage
        self.train_preds, self.train_targs, self.train_probs = [], [], []
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Store predictions for metrics
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)
        
        self.val_preds.extend(preds.detach().cpu().numpy().tolist())
        self.val_targs.extend(y.detach().cpu().numpy().tolist())
        self.val_probs.extend(probs.detach().cpu().numpy().tolist())
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def on_validation_epoch_end(self):
        """Compute validation metrics."""
        if self.val_targs:
            metrics = compute_all_metrics(
                self.val_targs,
                self.val_preds,
                np.array(self.val_probs),
                n_classes=5
            )
            
            # Log key metrics
            self.log("val_qwk", metrics['qwk'], prog_bar=True, on_epoch=True)
            self.log("val_acc", metrics['accuracy'], prog_bar=False, on_epoch=True)
            self.log("val_f1", metrics['macro_f1'], prog_bar=False, on_epoch=True)
            
            if 'roc_auc_ovr' in metrics:
                self.log("val_roc_auc", metrics['roc_auc_ovr'], prog_bar=False, on_epoch=True)
            
            # Log per-class metrics
            for i in range(5):
                self.log(f"val_precision_class_{i}", metrics[f'precision_class_{i}'], on_epoch=True)
                self.log(f"val_recall_class_{i}", metrics[f'recall_class_{i}'], on_epoch=True)
        
        # Reset storage
        self.val_preds, self.val_targs, self.val_probs = [], [], []
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.lr_scheduler is None:
            return optimizer
        
        if self.hparams.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.lr * 0.01
            )
        elif self.hparams.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.trainer.max_epochs // 3,
                gamma=0.1
            )
        elif self.hparams.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.lr_scheduler}")
        
        if self.hparams.lr_scheduler == 'plateau':
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_qwk'
                }
            }
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }


def compute_class_counts(fold_csv, split='train'):
    """Compute class counts from fold CSV."""
    import pandas as pd
    df = pd.read_csv(fold_csv)
    df = df[(df["label"].isin([0, 1, 2, 3, 4])) & (df["is_valid"] == True)]
    if "split" in df.columns:
        df = df[df["split"] == split]
    counts = df['label'].value_counts().sort_index().values
    # Ensure all 5 classes are represented
    full_counts = np.zeros(5, dtype=np.int64)
    for i, count in enumerate(counts):
        if i < 5:
            full_counts[i] = count
    return full_counts


def make_loaders(fold_csv, img_size=512, batch_size=16, num_workers=0, remove_borders=True):
    """Create train and validation dataloaders."""
    ds_tr = DRDataset(fold_csv, split="train", tfm=get_train_tf(img_size, remove_borders=remove_borders))
    ds_va = DRDataset(fold_csv, split="val", tfm=get_val_tf(img_size, remove_borders=remove_borders))
    
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dl_tr, dl_va


def main():
    ap = argparse.ArgumentParser(description="Train DR classification model")
    
    # Data arguments
    ap.add_argument("--fold_csv", required=True, help="Path to fold CSV file")
    ap.add_argument("--img_size", type=int, default=512, help="Image size")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size")
    ap.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    ap.add_argument("--remove_borders", action='store_true', default=True, help="Remove black borders")
    
    # Model arguments
    ap.add_argument("--model", default="efficientnet_b3", help="Model architecture")
    ap.add_argument("--freeze_backbone", action='store_true', help="Freeze backbone initially")
    ap.add_argument("--unfreeze_epoch", type=int, default=None, help="Epoch to unfreeze backbone")
    
    # Training arguments
    ap.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    ap.add_argument("--lr_scheduler", choices=['cosine', 'step', 'plateau', 'none'], default='cosine', help="LR scheduler")
    ap.add_argument("--warmup_epochs", type=int, default=0, help="Warmup epochs")
    
    # Loss arguments
    ap.add_argument("--loss", choices=['ce', 'weighted_ce', 'focal', 'label_smoothing'], default='ce', help="Loss function")
    ap.add_argument("--use_class_weights", action='store_true', help="Compute class weights from data")
    ap.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # Other arguments
    ap.add_argument("--monitor", default="val_qwk", help="Metric to monitor for checkpointing")
    ap.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = ap.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Compute class counts if needed
    class_counts = None
    if args.use_class_weights or args.loss in ['weighted_ce', 'focal']:
        class_counts = compute_class_counts(args.fold_csv, split='train')
        print(f"Class counts: {class_counts}")
    
    # Create dataloaders
    dl_tr, dl_va = make_loaders(
        args.fold_csv,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.remove_borders
    )
    
    # Filter loss kwargs based on loss type
    loss_kwargs = {}
    if args.loss == 'focal':
        loss_kwargs['gamma'] = args.focal_gamma
    elif args.loss == 'label_smoothing':
        loss_kwargs['label_smoothing'] = getattr(args, 'label_smoothing', 0.1)
    
    # Create model
    model = DRModule(
        model_name=args.model,
        lr=args.lr,
        img_size=args.img_size,
        loss_type=args.loss,
        class_counts=class_counts,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
        lr_scheduler=args.lr_scheduler if args.lr_scheduler != 'none' else None,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        **loss_kwargs
    )
    
    # Callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            monitor=args.monitor,
            mode="max",
            save_top_k=1,
            filename=f"{args.model}-s{args.img_size}-{{epoch:02d}}-{{{args.monitor}:.4f}}",
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step'),
        EarlyStopping(
            monitor=args.monitor,
            mode="max",
            patience=args.patience,
            verbose=True
        )
    ]
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        precision="32-true",
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=callbacks,
        deterministic=True
    )
    
    trainer.fit(model, dl_tr, dl_va)


if __name__ == "__main__":
    main()
