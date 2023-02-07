# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU
import os

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class XrayDataset(Dataset):
    def __init__(self, data_dir, split="train", width=224, height=224, transform=None):
        super(XrayDataset, self).__init__()
        self.data_dir = data_dir
        self.split = split
        self.width = width
        self.height = height
        self.transform = transform
        self.img_paths = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        pass


def split_dataset(
        annotation_dir="/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/Three_ROI_280_identities"):
    all_files, label_files, ID_labels = os.listdir(annotation_dir), [], {}

    for _file in all_files:
        if _file.endswith(".txt"):
            label_files.append(_file)
            ID = _file.split("_")[0]
            if ID in ID_labels.keys():
                ID_labels[ID].append(_file)
            else:
                ID_labels[ID] = [_file, ]

    # Printing out the
    with open("results/data_stat.csv", "w") as fp:
        print("img_count,id_count", file=fp)
        for i in range(1, 15):
            count = 0
            for key in ID_labels.keys():
                if len(ID_labels[key]) == i:
                    count += 1
            print(f"{i},{count}", file=fp)







if __name__ == "__main__":
    ds = XrayDataset(
        data_dir="/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/Annotations_T1-T5_280_identities",
        split="train")

    split_dataset()
