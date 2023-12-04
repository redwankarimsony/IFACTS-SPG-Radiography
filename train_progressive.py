#       train_progressive.py
#       Created by Redwan Sony (sonymd) at 11/22/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from configs.config_progressive import ConfigClass
from datasets import get_progressive_datasets
from models import build_model, build_head
from utils import (get_gpu_with_least_memory_over_period,
                   fuse_features_with_norm, get_verification_pairs, getTMRatFMR, roc_curve, calculate_eer)
from tqdm import tqdm

# Setting the precision
torch.set_float32_matmul_precision('high')


class ProgressiveLightningTrainer(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = build_model(model_arch=cfg.model_arch, embedding_size=cfg.embedding_size)
        self.head = build_head(head_type=cfg.loss_head, embedding_size=cfg.embedding_size, class_num=cfg.num_classes)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, images, labels):
        embeddings, norms = self.model(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100  # ignore_index
        return cos_thetas, norms, embeddings, labels

    def training_step(self, batch, batch_idx):
        # images, labels, file_names, file_infos = batch
        images, labels = batch
        cos_thetas, norms, embeddings, labels = self(images, labels)

        loss = self.loss_func(cos_thetas, labels)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'],
                 on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_loss', loss, batch_size=images.shape[0], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # images, labels, file_names, file_infos = batch
        images, labels = batch
        embeddings, norms = self.model(images)

        flipped_images = torch.flip(images, dims=[3])
        flipped_embeddings, flipped_norms = self.model(flipped_images)
        stacked_embeddings = torch.stack([embeddings, flipped_embeddings], dim=0)
        stacked_norms = torch.stack([norms, flipped_norms], dim=0)
        embeddings, norms = fuse_features_with_norm(stacked_embeddings, stacked_norms)

        # # dp requires the tensor to be cuda
        # return {
        #     'output': embeddings.to('cpu'),
        #     'norm': norms.to('cpu'),
        #     'target': labels.to('cpu'),
        #     'file_name': file_names}

        return {
            'output': embeddings,
            'norm': norms,
            'target': labels}

    def validation_epoch_end(self, outputs):
        embeddings = torch.cat([x['output'] for x in outputs], dim=0)
        norms = torch.cat([x['norm'] for x in outputs], dim=0)
        labels = torch.cat([x['target'] for x in outputs], dim=0)
        # file_names = [x['file_name'] for x in outputs]
        # file_names = [item for sublist in file_names for item in sublist]

        # verification
        pairs = get_verification_pairs(labels)
        pairs = np.array(pairs)

        y_preds, y_trues = [], []
        for pair in tqdm(pairs):
            emb1, norm1 = embeddings[pair[0]], norms[pair[0]]
            emb2, norm2 = embeddings[pair[1]], norms[pair[1]]

            # Find the cosine similarity between the embeddings
            cos_sim = torch.dot(emb1, emb2) / (norm1 * norm2)

            # Append the cosine similarity to the list of predictions
            y_preds.append(cos_sim.item())
            y_trues.append(pair[2])

        # Convert the lists to numpy arrays
        y_preds, y_trues = np.array(y_preds), np.array(y_trues)

        # Calculate the TMR at FMR = 0.01
        fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)
        tmr, threshold = getTMRatFMR(fpr, tpr, thresholds, target_fmr=0.02)

        # Calculate the EER
        eer = calculate_eer(y_trues, y_preds)

        # # Calculate the TMR, FMR, and EER
        # tmr, fmr, eer = calculate_metrics(y_trues, y_preds)

        # Log the metrics
        self.log("tmr_@_fmr", tmr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate,
                               weight_decay=self.cfg.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_callbacks(cfg):
    early_stopper_callback = EarlyStopping(monitor="tmr_@_fmr", min_delta=0.00, patience=20, verbose=False, mode="max")

    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="tmr_@_fmr", mode="max",
                                          dirpath=cfg.checkpoint_dir,
                                          filename=f"{cfg.model_arch}+{cfg.loss_head}+{cfg.split_mode}+{cfg.box_preset}"
                                                   +"{epoch:02d}+{tmr_@_fmr:.4f}",
                                          )
    return [early_stopper_callback, checkpoint_callback]


def main(cfg):
    # Load the trainer model
    classifier = ProgressiveLightningTrainer(cfg)

    # Load the datasets
    ds_train, ds_valid, ds_test, dl_train, dl_valid, dl_test = get_progressive_datasets(cfg)

    # Load the logger
    logger = TensorBoardLogger(save_dir=os.path.join(cfg.results_dir, cfg.split_mode),
                               name=cfg.box_preset,
                               version=cfg.model_arch)
    #
    # # Load the callbacks
    callbacks = get_callbacks(cfg)

    # Load the trainer
    trainer = Trainer(limit_train_batches=cfg.limit_train_batches,
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
    """ 
    To run this script:
        python train_progressive.py --model_arch efficientnet_b2 --box_preset t1-t5  --split_mode case_in_random
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
                        default=get_gpu_with_least_memory_over_period(10),
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
                        default="train_progressive",
                        help="Select the experiment name")

    parser.add_argument("--split_mode", type=str, default="case_in_test",
                        choices={"case_in_test", "case_in_train", "case_in_random"})
    # take a flag for horizontal flip
    parser.add_argument("--enable_flip", action="store_true", default=False)


    args = parser.parse_args()
    print(f'Lowest GPU usage: {args.gpu}')

    # Prepare the configuration
    cfg = ConfigClass(**vars(args))
    print(cfg)

    # Call the trainer function
    main(cfg)
    time.sleep(10)
