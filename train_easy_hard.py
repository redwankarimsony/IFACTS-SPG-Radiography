# train_easy_hard.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

import argparse
import torch
import pytorch_lightning as pl
import torchmetrics

from torch import optim, nn
from torch.utils.data import DataLoader
from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from configs.config_easy_hard import configClass
from dataset import XrayDataset
from models import get_model
from inference import select_the_best_model

torch.set_float32_matmul_precision("high")


class LightningTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(self.cfg)
        self.loss_func = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(dim=1)

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


def get_dataloader(cfg, split, difficulty, transform):
    ds = XrayDataset(cfg, split=split, difficulty=difficulty, use_cache=cfg.use_cache, transform=transform)
    dl = DataLoader(ds,
                    batch_size=cfg.batch_size,
                    shuffle=True if split=='train' else False,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    persistent_workers=True)
    return dl, ds


if __name__ == "__main__":
    """ To run this script:
        python train.py --model_arch resnet34 --gpu 7 --box_preset 3
    """
    parser = argparse.ArgumentParser(prog='IFACTS Experiment',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument("--model_arch", type=str, default="resnet34",
                        help="Select the model selection key.\nFor more look for cfr.model_arch in models.py file")
    parser.add_argument("--gpu", type=int, default=7,
                        help="Select which gpu you would like to use")
    parser.add_argument("--box_preset", type=int, default=3, choices={0, 1, 2, 3},
                        help="Select the bounding box configuration")
    args = parser.parse_args()

    cfg = configClass()
    cfg.model_arch = args.model_arch
    cfg.cuda_devices = [args.gpu, ]
    cfg.box_preset = args.box_preset
    cfg.make_results_dir()

    # split_dataset()
    # ds_train = XrayDataset(cfg, split="train", transform=cfg.tfms_train)
    # ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)

    # dl_train = DataLoader(ds_train,
    #                       batch_size=cfg.batch_size,
    #                       shuffle=True,
    #                       num_workers=cfg.num_workers,
    #                       pin_memory=cfg.pin_memory,
    #                       persistent_workers=True)

    # dl_valid = DataLoader(ds_valid,
    #                       batch_size=cfg.batch_size,
    #                       shuffle=False,
    #                       num_workers=cfg.num_workers,
    #                       pin_memory=cfg.pin_memory,
    #                       persistent_workers=True)

    dl_train_easy, ds_train_easy = get_dataloader(cfg=cfg, split='train', difficulty='easy', transform=cfg.tfms_train)
    dl_train_hard, ds_train_hard = get_dataloader(cfg=cfg, split='train', difficulty='hard', transform=cfg.tfms_train)
    dl_train_all, ds_train_all = get_dataloader(cfg=cfg, split='train', difficulty='all', transform=cfg.tfms_train)
    dl_valid, ds_valid = get_dataloader(cfg=cfg, split="valid", difficulty="all", transform=cfg.tfms_valid)

    classifier = LightningTrainer(cfg)

    early_stopper_callback = EarlyStopping(monitor="val_loss",
                                           min_delta=0.00,
                                           patience=60,
                                           verbose=False,
                                           mode="min")

    checkpoint_callback = ModelCheckpoint(save_top_k=5,
                                          monitor="val_acc_epoch",
                                          mode="max",
                                          dirpath=f"{cfg.checkpoints_dir}/box_{cfg.box_preset}/{cfg.model_arch}",
                                          filename=cfg.model_arch + "_{epoch:02d}-{val_acc_epoch:.4f}",
                                          )
    logger = loggers.TensorBoardLogger(save_dir=cfg.checkpoints_dir,
                                       name=f"box_{cfg.box_preset}",
                                       version=cfg.model_arch)

    # trainer = pl.Trainer(limit_train_batches=cfg.limit_train_batches,
    #                      max_epochs=int(cfg.max_epoch * 0.33),
    #                      accelerator="gpu",
    #                      devices=cfg.cuda_devices,
    #                      log_every_n_steps=25,
    #                      default_root_dir=cfg.checkpoints_dir,
    #                      logger=logger,
    #                      callbacks=[early_stopper_callback, checkpoint_callback], 
    #                      gradient_clip_val=0.5,
    #                      gradient_clip_algorithm="value")

    # trainer.fit(model=classifier, train_dataloaders=dl_train_easy, val_dataloaders=dl_valid)



    # # Resume the training process with hard set
    # best_model = select_the_best_model(save_dir=cfg.checkpoints_dir,
    #                                    model_arch=cfg.model_arch,
    #                                    box_preset=cfg.box_preset)
    
    # trainer = pl.Trainer(limit_train_batches=cfg.limit_train_batches,
    #                      max_epochs=int(cfg.max_epoch * 0.66),
    #                      accelerator="gpu",
    #                      devices=cfg.cuda_devices,
    #                      log_every_n_steps=25,
    #                      default_root_dir=cfg.checkpoints_dir,
    #                      logger=logger,
    #                      callbacks=[early_stopper_callback, checkpoint_callback], 
    #                      gradient_clip_val=0.5,
    #                      gradient_clip_algorithm="value")
    
    # trainer.fit(model=classifier, 
    #             train_dataloaders=dl_train_hard, 
    #             val_dataloaders=dl_valid,
    #             ckpt_path=best_model)
    


    # Resume the training process with all
    best_model = select_the_best_model(save_dir=cfg.checkpoints_dir,
                                       model_arch=cfg.model_arch,
                                       box_preset=cfg.box_preset)
    
    trainer = pl.Trainer(limit_train_batches=cfg.limit_train_batches,
                         max_epochs=cfg.max_epoch,
                         accelerator="gpu",
                         devices=cfg.cuda_devices,
                         log_every_n_steps=25,
                         default_root_dir=cfg.checkpoints_dir,
                         logger=logger,
                         callbacks=[early_stopper_callback, checkpoint_callback], 
                         gradient_clip_val=0.5,
                         gradient_clip_algorithm="value")
    
    trainer.fit(model=classifier, 
                train_dataloaders=dl_train_all, 
                val_dataloaders=dl_valid,
                ckpt_path=best_model)


    




