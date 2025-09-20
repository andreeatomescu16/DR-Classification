import argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from drlib.datasets import DRDataset
from drlib.transforms import get_train_tf, get_val_tf
from drlib.models import create_model
from drlib.metrics import quadratic_weighted_kappa

class DRModule(L.LightningModule):
    def __init__(self, model_name="efficientnet_b3", lr=1e-4, img_size=512):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, num_classes=5, pretrained=True)
        self.criterion = nn.CrossEntropyLoss()
        self.train_preds, self.train_targs = [], []
        self.val_preds, self.val_targs = [], []

    def forward(self, x): return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        self.train_targs.extend(y.detach().cpu().numpy().tolist())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if self.train_targs:
            kappa = quadratic_weighted_kappa(self.train_targs, self.train_preds)
            self.log("train_qwk", kappa, prog_bar=True, on_epoch=True)
        self.train_preds, self.train_targs = [], []


    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_preds.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        self.val_targs.extend(y.detach().cpu().numpy().tolist())
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        if self.val_targs:
            kappa = quadratic_weighted_kappa(self.val_targs, self.val_preds)
            # show in progress bar and save in logs
            self.log("val_qwk", kappa, prog_bar=True, on_epoch=True)
        self.val_preds, self.val_targs = [], []


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

def make_loaders(fold_csv, img_size=512, batch_size=16, num_workers=4):
    ds_tr = DRDataset(fold_csv, split="train", tfm=get_train_tf(img_size))
    ds_va = DRDataset(fold_csv, split="val",   tfm=get_val_tf(img_size))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return dl_tr, dl_va

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_csv", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model", default="efficientnet_b3")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    dl_tr, dl_va = make_loaders(args.fold_csv, args.img_size, args.batch_size, args.num_workers)
    model = DRModule(model_name=args.model, lr=args.lr, img_size=args.img_size)

    ckpt = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_qwk", mode="max", save_top_k=1, filename=f"{args.model}-s{args.img_size}-fold{{epoch:02d}}-{{val_qwk:.4f}}"
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",   # will pick 'mps' on Apple Silicon
        devices=1,
        precision="32-true",  # avoid CUDA AMP warnings on MPS/CPU
        log_every_n_steps=10,
        enable_progress_bar=True,
        callbacks=[ckpt],
    )
    trainer.fit(model, dl_tr, dl_va)

if __name__ == "__main__":
    main()
