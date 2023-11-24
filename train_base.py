# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

import argparse
import os
import torch
import pytorch_lightning as pl
import torchmetrics
import time

from torch import optim, nn
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from configs.config import ConfigClass
# from dataset import XrayDataset
from datasets import MXNetRecDataset
from models import get_model
from utils import get_gpu_with_least_memory_over_period
import numpy as np

torch.set_float32_matmul_precision("high")


class LightningTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningTrainer, self).__init__()
        self.save_hyperparameters()

        self.cfg = cfg

        # Model related attributes
        self.model = self._initialize_model()
        self.loss_func = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        # Metrics related attributes
        num_classes = self.cfg.num_classes
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _initialize_model(self):
        """Initialize the model based on the given configuration."""
        return get_model(self.cfg)

    def training_step(self, batch, batch_idx):
        # Extract data from batch
        X, y, codes = batch

        # Forward pass
        y_hat = self.model(X)
        preds = self.softmax(y_hat)

        # Compute loss
        loss = self.loss_func(preds, y)

        # Log metrics and results
        self._log_training_metrics(preds, y, loss, len(X))

        return loss

    def _log_training_metrics(self, preds, y, loss, batch_size):
        """Log training metrics to tensorboard and console."""
        self.train_acc(preds, y)
        self.log("train_loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("train_acc_step", self.train_acc, batch_size=batch_size, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # Extract data from batch
        X, y, codes = batch

        # Forward pass and compute loss
        y_hat = self.model(X)
        preds = self.softmax(y_hat)
        loss = self.loss_func(preds, y)

        # Log metrics and results
        self._log_validation_metrics(preds, y, loss, len(X))

        return loss

    def _log_validation_metrics(self, preds, y, loss, batch_size):
        """Log validation metrics to tensorboard and console."""
        self.val_acc(preds, y)
        self.log("val_loss", loss, batch_size=batch_size, sync_dist=True)
        self.log("val_acc_step", self.val_acc, batch_size=batch_size, sync_dist=True)

    def configure_optimizers(self):
        """Define the optimizer for the model."""
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        return optimizer

    def training_epoch_end(self, train_step_output):
        """Operations to perform at the end of each training epoch."""
        train_accuracy = self.train_acc.compute()
        self.log("train_acc_epoch", train_accuracy, sync_dist=True)

    def validation_epoch_end(self, val_step_output):
        """Operations to perform at the end of each validation epoch."""
        val_accuracy = self.val_acc.compute()
        self.log("val_acc_epoch", val_accuracy, sync_dist=True)


def get_datasets(cfg):
    # Define the dataset paths
    ds_path_train = f"{cfg.dataset_dir}/train.rec"
    ds_path_valid = f"{cfg.dataset_dir}/valid.rec"

    if cfg.use_mxrecord:
        # Create an instance of the MXNetRecDataset
        ds_train = MXNetRecDataset(rec_path=ds_path_train, transform=cfg.tfms_train)
        ds_valid = MXNetRecDataset(rec_path=ds_path_valid, transform=cfg.tfms_valid)
    else:
        # # Create an instance of the XrayDataset
        # ds_train = XrayDataset(cfg, split="train", transform=cfg.tfms_train)
        # ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)
        pass

    # Create the dataloaders for train and valid
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

    return ds_train, ds_valid, dl_train, dl_valid


def get_logger(cfg):
    logger = loggers.TensorBoardLogger(save_dir=cfg.results_dir,
                                       name=f"{cfg.box_preset}",
                                       version=cfg.model_arch)
    return logger


def get_callbacks(cfg):
    early_stopper_callback = EarlyStopping(monitor="val_loss",
                                           min_delta=0.00,
                                           patience=40,
                                           verbose=False,
                                           mode="min")

    checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                          monitor="val_acc_epoch",
                                          mode="max",
                                          dirpath=os.path.join(cfg.results_dir, cfg.box_preset, cfg.model_arch),

                                          filename=cfg.model_arch + "_" + cfg.box_preset + "_{epoch:02d}-{val_acc_epoch:.4f}",
                                          )
    return [early_stopper_callback, checkpoint_callback]


def main(cfg):
    # Load teh classifier trainer
    classifier = LightningTrainer(cfg)

    # Load the datasets
    ds_train, ds_valid, dl_train, dl_valid = get_datasets(cfg)

    # Load the logger
    logger = get_logger(cfg)

    # Load the callbacks
    callbacks = get_callbacks(cfg)

    # Load the trainer
    trainer = pl.Trainer(limit_train_batches=cfg.limit_train_batches,
                         max_epochs=cfg.max_epoch,
                         accelerator="gpu",
                         devices=cfg.cuda_devices,
                         log_every_n_steps=50,
                         default_root_dir=os.path.join(cfg.results_dir,
                                                       cfg.box_preset,
                                                       cfg.model_arch),
                         logger=logger,
                         callbacks=callbacks,
                         gradient_clip_val=0.5,
                         gradient_clip_algorithm="value")

    # Fit the model
    trainer.fit(model=classifier,
                train_dataloaders=dl_train,
                val_dataloaders=dl_valid)


if __name__ == "__main__":
    """ To run this script:
        python train.py --model_arch resnet34 --gpu 7 --box_preset 3
    """

    parser = argparse.ArgumentParser(prog='IFACTS Experiment',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument("--model_arch",
                        type=str,
                        default="resnet34",
                        help="Select the model selection key.\nFor more look for cfr.model_arch in models.py file")

    parser.add_argument("--gpu",
                        type=int,
                        default=get_gpu_with_least_memory_over_period(20),
                        help="Select which gpu you would like to use")

    parser.add_argument("--box_preset",
                        type=str,
                        default='t1-t5',
                        choices={'t1-t5',
                                 'clavicle-only',
                                 'complete-vertebrae',
                                 'whole'},
                        help="Select the bounding box configuration")

    parser.add_argument("--experiment_name",
                        type=str,
                        default="base_experiment",
                        help="Select the experiment name")

    args = parser.parse_args()

    print(f'Lowest GPU usage: {args.gpu}')

    cfg = ConfigClass(**vars(args))
    print(cfg)

    main(cfg)

    torch.cuda.empty_cache()
    time.sleep(30)

    # Write the confirmation that code has finished running
    with open(os.path.join(cfg.results_dir, "summary", "done.txt"), "w") as f:
        f.write(f"{cfg.experiment_name} {cfg.model_arch} {cfg.box_preset} done")
