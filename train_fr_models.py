import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import get_backbone, build_head



class FaceRecognitionBasedModel(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.backbone = get_backbone(cfg)
        self.head = build_head(cfg.head_type, 
                               embedding_size=cfg.embedding_size, 
                               class_num=32000, 
                               m=cfg.m, 
                               t_alpha=cfg.t_alpha, 
                               h=cfg.h, 
                               s=cfg.s, )
        
        self.loss_func = nn.CrossEntropyLoss()

        

    def forward(self, images, labels):
        embeddings, norms = self.backbone(images)
        cos_thetas = self.head(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, bad_grad = cos_thetas
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        return cos_thetas, norms, embeddings, labels
    

    def training_step(self, batch, batch_idx):
        # Extract data from batch
        X, y, codes = batch
        cos_thetas, norms, embeddings, labels = self(X, y)
        loss_train = self.loss_func(cos_thetas, labels)
        lr = self.get_current_lr()

        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)    
        self.log("train_loss", loss_train, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_train




    
    






    