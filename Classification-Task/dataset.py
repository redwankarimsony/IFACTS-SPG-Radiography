# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU
import os
import os.path as osp
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from configs.config import cfg
import math


class XrayDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        super(XrayDataset, self).__init__()
        self.img_dir = cfg.img_dir
        self.annot_dir = cfg.annot_dir
        self.split = split
        self.width = cfg.width
        self.height = cfg.height
        self.transform = transform
        self.df = pd.read_csv(osp.join(cfg.results_dir, "data_splits.csv"))
        self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
        self.labels = {}

        self._load_labels()

        print(self.labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, f"{self.df.iloc[idx, 0]}.png")
        annot_path = osp.join(self.img_dir, f"{self.df.iloc[idx, 0]}.txt")


    def _load_labels(self):
        temp = pd.read_csv(f"{cfg.results_dir}/labels_mapping.csv")
        for idx, row in temp.iterrows():
            self.labels[row["ID"]] = row["label_idx"]


def split_dataset(annot_dir=cfg.annot_dir):
    all_files, label_files, ID_labels = os.listdir(annot_dir), [], {}
    train_files, valid_files, IDs = [], [], set({})

    # Listing all the label files
    for _file in all_files:
        if _file.endswith(".txt"):
            label_files.append(_file)
            ID = _file.split("_")[0]
            IDs.add(ID)
            if ID in ID_labels.keys():
                ID_labels[ID].append(_file)
            else:
                ID_labels[ID] = [_file, ]

    # Printing out the data statistics
    with open(f"{cfg.results_dir}/data_stat.csv", "w") as fp:
        print("img_count,id_count", file=fp)
        for i in range(1, 15):
            count = 0
            for key in ID_labels.keys():
                _files = ID_labels[key]
                if len(_files) == i:
                    count += 1
                print(f"{i},{count}", file=fp)

    # splitting up the train and validation images.
    for ID in IDs:
        _files = ID_labels[ID]
        if len(_files) == 1:
            train_files.extend(_files)
        elif len(_files) == 2:
            train_files.append(_files[0])
            valid_files.append(_files[1])
        elif len(_files) == 3:
            train_files.extend(_files[:2])
            valid_files.extend(_files[2:])
        else:
            num_t = int(math.ceil(len(_files) * (1 - cfg.valid_ratio)))
            train_files.extend(_files[:num_t])
            valid_files.extend(_files[num_t:])

    print(f"Total Images: {len(all_files)}",
          f"\nTrain Images: {len(train_files)}",
          f"\nValid Images: {len(valid_files)}")

    # Making the data frame for later reference
    image_ids, ids, splits = [], [], []
    for _file in train_files:
        image_ids.append(_file.replace(".txt", ""))
        ids.append(_file.split("_")[0]),
        splits.append("train")

    for _file in valid_files:
        image_ids.append(_file.replace(".txt", ""))
        ids.append(_file.split("_")[0]),
        splits.append("valid")

    df = pd.DataFrame({"image_id": image_ids,
                       "id": ids,
                       "split": splits})
    csv_path = osp.join(cfg.results_dir, "data_splits.csv")
    df.to_csv(csv_path, index=False)
    print(f"Splits written at : ./{csv_path}")

    # Writing out the labels mapping to index
    IDs = sorted(list(IDs))
    with open(f"{cfg.results_dir}/labels_mapping.csv", "w") as fp:
        print("ID,""label_idx", file=fp)
        for idx, ID in enumerate(IDs):
            print(f"{ID}, {idx}", file=fp)


if __name__ == "__main__":
    split_dataset()
    ds = XrayDataset(cfg, split="valid", transform=None)
    print(len(ds))
