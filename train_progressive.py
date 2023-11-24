#       train_progressive.py
#       Created by Redwan Sony (sonymd) at 11/22/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import time
from typing import Any

import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from models import get_fr_model
from configs.config_progressive import ConfigClass
from datasets.progressive import get_progressive_datasets
from utils import get_gpu_with_least_memory_over_period


class ProgressiveLightningTrainer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_fr_model(cfg)
        self.loss_func = nn.CrossEntropyLoss()

        if self.hparams.start_from_model_statedict:
            ckpt = torch.load(self.hparams.start_from_model_statedict)
            self.model.load_state_dict({key.replace('model.', ''): val
                                        for key, val in ckpt['state_dict'].items() if 'model.' in key})

    def forward(self, images, labels):
        embeddings, norms = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        return cos_thetas, norms, embeddings, labels

    def training_step(self, batch, batch_idx):
        images, labels = batch
        cos_thetas, norms, embeddings, labels = self(images, labels)
        loss = self.loss_func(cos_thetas, labels)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        cos_thetas, norms, embeddings, labels = self(images, labels)
        loss = self.loss_func(cos_thetas, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr,
                               weight_decay=self.cfg.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass


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
                                          filename=f"{cfg.model_arch}+{cfg.loss_head}" + "_" +
                                                   cfg.box_preset + "_{epoch:02d}-{val_acc_epoch:.4f}",
                                      )
    return [early_stopper_callback, checkpoint_callback]


def main():
    # Load the trainer model
    trainer = ProgressiveLightningTrainer(cfg)

    # Load the datasets
    ds_train, ds_valid, ds_test, dl_train, dl_valid, dl_test = get_progressive_datasets(cfg)

    # Load the logger
    logger = TensorBoardLogger(save_dir=cfg.results_dir,
                               name=cfg.experiment_name,
                               version=cfg.model_arch)

    # Load the callbacks
    callbacks = get_callbacks(cfg)




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

    # Prepare the configuration
    cfg = ConfigClass(**vars(args))
    print(cfg)

    # Call the trainer function
    time.sleep(10)


