# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU


import math
import os
import os.path as osp
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as tf
from configs.config import cfg

torch.manual_seed(1455)
plt.rcParams["font.size"] = 12
plt.rcParams["lines.linewidth"] = 2


class XrayDataset(Dataset):
    def __init__(self, cfg, split="train", transform=None):
        super(XrayDataset, self).__init__()
        self.img_dir = cfg.img_dir
        self.annot_dir = cfg.annot_dir
        self.split = split
        self.width = cfg.width
        self.height = cfg.height
        self.box_preset = cfg.box_preset
        self.transform = transform
        self.df = pd.read_csv(osp.join(cfg.results_dir, "data_splits.csv"))
        self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)
        self.labels = {}

        self._load_labels()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, f"{self.df.iloc[idx, 0]}.png")
        annot_path = osp.join(self.annot_dir, f"{self.df.iloc[idx, 0]}.txt")

        df = pd.read_csv(annot_path, delimiter=" ", names=["label", "x", "y", "w", "h"], header=None)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_mask = np.zeros_like(img)
        H, W = img.shape

        if cfg.box_preset != 3:
            _, x, y, w, h = df.iloc[cfg.box_preset, :]
        else:
            _, x, y, w, h = 4, 0.5, 0.5, 1, 1

        xmin, ymin, xmax, ymax = self.bbox2points([x, y, w, h], H, W,
                                                  box_preset=self.box_preset,
                                                  take_square=False)
        img_mask[ymin:ymax, xmin:xmax] = img[ymin:ymax, xmin:xmax]

        print(xmin, ymin, xmax, ymax, img.shape, self.df.iloc[idx, 0])



        # img = torch.tensor(img / 255.0)
        # if self.transform:
        #     img = self.transform(img)
        # print(img)
        return [img, img_mask], " ", self.df.iloc[idx, 0]

    def _load_labels(self):
        temp = pd.read_csv(f"{cfg.results_dir}/labels_mapping.csv")
        for idx, row in temp.iterrows():
            self.labels[row["ID"]] = row["label_idx"]

    @staticmethod
    def bbox2points(bbox, H, W, box_preset, take_square=True):
        """
        From bounding box yolo format
        to corner points cv2 rectangle
        """
        x, y, w, h = bbox
        if take_square:
            if box_preset == 0:
                xmin = int((x - (h / 2)) * W)
                xmax = int((x + (h / 2)) * W)
                ymin = int((y - (h / 2)) * H)
                ymax = int((y + (h / 2)) * H)
            elif box_preset == 1:
                xmin = 0
                xmax = W
                ymin = int((y - (h / 2)) * H)
                ymax = int((y + (h / 2)) * H)
            elif box_preset == 2:
                xmin = int((x - (w / 2)) * W)
                xmax = int((x + (w / 2)) * W)
                ymin = int((y - (w / 2)) * H)
                ymax = int((y + (w / 2)) * H)

        else:
            xmin = int((x - (w / 2)) * W)
            xmax = int((x + (w / 2)) * W)
            ymin = int((y - (h / 2)) * H)
            ymax = int((y + (h / 2)) * H)
        return xmin, ymin, xmax, ymax


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

    df = pd.DataFrame({"image_id": image_ids, "id": ids, "split": splits})
    csv_path = osp.join(cfg.results_dir, "data_splits.csv")
    df.to_csv(csv_path, index=False)
    print(f"Splits written at : ./{csv_path}")

    # Writing out the labels mapping to index
    IDs = sorted(list(IDs))
    with open(f"{cfg.results_dir}/labels_mapping.csv", "w") as fp:
        print("ID,""label_idx", file=fp)
        for idx, ID in enumerate(IDs):
            print(f"{ID}, {idx}", file=fp)







transfroms = torch.nn.Sequential(
    tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
if __name__ == "__main__":
    split_dataset()
    ds = XrayDataset(cfg, split="valid", transform=None)
    print(len(ds))
    [img, img_mask], label, file_id = ds[0]
    print(img.shape)

    print(img)
    print((img + img_mask) // 2)

    fig = plt.figure(figsize=(15, 15), dpi=200)
    fig.add_subplot(2, 2, 1)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Original Image")

    fig.add_subplot(2, 2, 2)
    plt.imshow(img_mask, cmap=plt.cm.gray)
    plt.title("Masked Image")

    fig.add_subplot(2,2,3)
    plt.imshow((img+img_mask)//2, cmap=plt.cm.gray)
    plt.title("Average Channel")

    fig.add_subplot(2,2,4)
    plt.imshow(np.maximum(img, img_mask), cmap=plt.cm.gray)
    plt.title("Reconstruction")
    fig.suptitle(f"Comparison of Different Channels\n{file_id}")
    plt.show()