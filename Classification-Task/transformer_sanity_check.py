from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torchvision.transforms as tf
from torch.nn import Sequential
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
from torchvision.models import swin_t, swin_s, swin_b
import pytorch_lightning as pl
import torchmetrics

from pytorch_lightning import loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint








class LightningTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = swin_t(weights = Swin_T_Weights.DEFAULT, progress =True)
        self.model.head = nn.Linear(self.model.head.in_features, 8)
        self.loss_func = nn.CrossEntropyLoss()

        self.softmax = nn.Softmax(dim=1)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        preds = self.softmax(y_hat)
        loss = self.loss_func(preds, y)
        batch_size = len(X)
        # logging the results
        self.train_acc(preds, y)
        self.log("train_loss", loss, batch_size=batch_size, logger=True)
        self.log("train_acc_step", self.train_acc, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        X, y = batch
        y_hat = self.model(X)
        preds = self.softmax(y_hat)
        loss = self.loss_func(preds, y)
        batch_size = len(X)

        # logging the results
        self.val_acc(preds, y)
        self.log("val_loss", loss, batch_size=batch_size, logger=True)
        self.log("val_acc_step", self.val_acc, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer

    def training_epoch_end(self, train_step_output):
        self.log("train_acc_epoch", self.train_acc.compute(), sync_dist=True)

    def validation_epoch_end(self, val_step_output):
        self.log("val_acc_epoch", self.val_acc.compute(), sync_dist=True)


if __name__ =="__main__":

    model_arch = "swin_t"

    tfms_valid = tf.Compose([tf.Resize(512),
                                tf.CenterCrop(512),
                                tf.ToTensor(),
                                # tf.RandomRotation(degrees=5),
                                # tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.6),
                                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    ds = ImageFolder("/home/sonymd/Downloads/natural_images_classification", transform=tfms_valid )
    ds_train, ds_valid = random_split(ds, [6000, 899])

    dl_train = DataLoader(ds_train,
                          batch_size=8,
                          shuffle=True,
                          num_workers=8,
                          pin_memory=True,
                          persistent_workers=True)

    dl_valid = DataLoader(ds_valid,
                          batch_size=8,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                          persistent_workers=True)
    classifier = LightningTrainer()

    early_stopper_callback = EarlyStopping(monitor="val_loss",
                                           min_delta=0.00,
                                           patience=30,
                                           verbose=False,
                                           mode="min")

    checkpoint_callback = ModelCheckpoint(save_top_k=10,
                                          monitor="val_acc_epoch",
                                          mode="max",
                                          dirpath=f"vit_checkpoints/{model_arch}",
                                          filename=model_arch + "_{epoch:02d}-{val_acc_epoch:.4f}",
                                          )
    logger = loggers.TensorBoardLogger(save_dir="vit_checkpoints",
                                       name=f"SWIN",
                                       version=model_arch)

    trainer = pl.Trainer(limit_train_batches=20,
                         max_epochs=20,
                         accelerator="gpu",
                         devices=[0,1],
                         log_every_n_steps=10,
                         default_root_dir="vit_checkpoints",
                         logger=logger,
                         callbacks=[early_stopper_callback, checkpoint_callback])

    trainer.fit(model=classifier, train_dataloaders=dl_train, val_dataloaders=dl_valid)