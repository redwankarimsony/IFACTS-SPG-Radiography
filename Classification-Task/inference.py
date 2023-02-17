#       inference.py
#       Created by Redwan Sony (sonymd) at 2/5/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import torch
import pytorch_lightning as pl
from dataset import XrayDataset



from configs.config import cfg



def inference(cfg):
    ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)
    print(len(ds_valid))








if __name__ == "__main__":
    inference(cfg)
