import torch
from torch import optim, nn, utils
import pytorch_lightning as pl
from configs.config import cfg
from models import get_model
from dataset import XrayDataset, split_dataset, transfroms


class LightningTrainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(self.cfg)
        self.loss_func = nn.CrossEntropyLoss()
        # print(self.model)

    def training_step(self, batch, batch_idx):
        X, y, codes = batch
        y_hat = self.model(X)
        loss = self.loss_func(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y, codes = batch
        y_hat = self.model(X)
        loss = self.loss_func(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        return optimizer

    # def configure_optimizers(self):
    #     pass


if __name__ == "__main__":
    split_dataset()
    ds_train = XrayDataset(cfg, split="train", transform=transfroms)
    ds_valid = XrayDataset(cfg, split="valid", transform=transfroms)

    dl_train = utils.data.DataLoader(ds_train,
                                     batch_size=cfg.batch_size,
                                     shuffle=True,
                                     num_workers=cfg.num_workers,
                                     pin_memory=cfg.pin_memory)

    dl_valid = utils.data.DataLoader(ds_valid,
                                     batch_size=cfg.batch_size,
                                     shuffle=True,
                                     num_workers=cfg.num_workers,
                                     pin_memory=cfg.pin_memory)
    classifier = LightningTrainer(cfg)

    trainer = pl.Trainer(limit_train_batches=100,
                         max_epochs=10,
                         accelerator="gpu",
                         devices=2)

    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_valid)
