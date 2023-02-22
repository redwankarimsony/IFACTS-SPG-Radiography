#       inference.py
#       Created by Redwan Sony (sonymd) at 2/5/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import XrayDataset
from train import LightningTrainer
from sklearn.metrics import confusion_matrix



from configs.config import cfg



def inference(cfg):
    ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)
    dl_valid = DataLoader(ds_valid,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          persistent_workers=True)
    print(len(ds_valid))
    model = LightningTrainer.load_from_checkpoint("saved_checkpoints/box_2/densenet121/densenet121_epoch=308-val_acc_epoch=0.7848.ckpt")

    predictions = []
    gt = []
    ml_model = model.model.to("cuda:7")
    with torch.no_grad():
        for idx, batch in enumerate(dl_valid):
            X, y, codes = batch
            print(y.shape)
            y_hat = ml_model(X.to("cuda:7"))
            preds = model.softmax(y_hat)
            predictions.append(preds.cpu())
            gt.extend(y.tolist())
        all_results = torch.vstack(predictions)
        final_predict = torch.argmax(all_results, axis=1).tolist()
        print(len(final_predict), len(gt))


        print(final_predict[:5])
        print(gt[:5])


        conf_matrix = confusion_matrix(gt, final_predict)
        print(conf_matrix[:45, :45])








if __name__ == "__main__":
    inference(cfg)
