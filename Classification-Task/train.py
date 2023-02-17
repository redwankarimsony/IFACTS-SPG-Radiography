# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU
import torch
import pytorch_lightning as pl
import torchmetrics

from torch import optim, nn
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from configs.config import cfg
from dataset import XrayDataset
from models import get_model

torch.set_float32_matmul_precision(cfg.cuda_precision)


class LightningTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(self.cfg)
        self.loss_func = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.cfg.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.cfg.num_classes)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        X, y, codes = batch
        y_hat = self.model(X)
        preds = self.softmax(y_hat)
        loss = self.loss_func(preds, y)
        batch_size = len(X)
        # logging the results
        self.train_acc(preds, y)
        self.log("train_loss", loss, batch_size=batch_size, logger=True, sync_dist=True)
        self.log("train_acc_step", self.train_acc, batch_size=batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y, codes = batch
        y_hat = self.model(X)
        preds = self.softmax(y_hat)
        loss = self.loss_func(preds, y)
        batch_size = len(X)

        # logging the results
        self.val_acc(preds, y)
        self.log("val_loss", loss, batch_size=batch_size, logger=True, sync_dist=True)
        self.log("val_acc_step", self.val_acc, batch_size=batch_size, sync_dist=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        return optimizer

    def training_epoch_end(self, train_step_output):
        self.log("train_acc_epoch", self.train_acc.compute(), sync_dist=True)

    def validation_epoch_end(self, val_step_output):
        self.log("val_acc_epoch", self.val_acc.compute(), sync_dist=True)


if __name__ == "__main__":
    # split_dataset()
    ds_train = XrayDataset(cfg, split="train", transform=cfg.tfms_train)
    ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)

    dl_train = DataLoader(ds_train,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          persistent_workers=True)

    dl_valid = DataLoader(ds_valid,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          persistent_workers=True)
    classifier = LightningTrainer(cfg)

    early_stopper_callback = EarlyStopping(monitor="val_loss",
                                           min_delta=0.00,
                                           patience=50,
                                           verbose=False,
                                           mode="min")

    checkpoint_callback = ModelCheckpoint(save_top_k=10,
                                          monitor="val_acc_epoch",
                                          mode="max",
                                          dirpath=f"saved_checkpoints/box_{cfg.box_preset}/{cfg.model_arch}",
                                          filename=cfg.model_arch + "_{epoch:02d}-{val_acc_epoch:.4f}",
                                          )
    logger = loggers.TensorBoardLogger(save_dir="saved_checkpoints",
                                       name=f"box_{cfg.box_preset}",
                                       version=cfg.model_arch)

    trainer = pl.Trainer(limit_train_batches=cfg.limit_train_batches,
                         max_epochs=cfg.max_epoch,
                         accelerator="gpu",
                         devices=cfg.cuda_devices,
                         log_every_n_steps=10,
                         default_root_dir="saved_checkpoints",
                         logger=logger,
                         callbacks=[early_stopper_callback, checkpoint_callback])

    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_valid)
