import os.path as osp
from glob import glob
from itertools import combinations
from random import sample

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data


from .SiameseDatasetIFACTS_v2 import pil_loader, cropImage, squareCROP


class SiameseDatasetChextXray14(data.Dataset):
    def __init__(self, img_dir, annot_dir, phase, data_handling="balanced", n_channels=3, n_samples=100000,
                 transform=None, crop_box=True):
        super().__init__()

        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.phase = phase
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.transform = transform
        self.data_handling = data_handling
        self.crop_box = crop_box
        self.image_pairs = []
        self.pos_pairs = []
        self.neg_pairs = []

        if self.phase == 'training':
            self.annot_dir = osp.join(self.annot_dir, "train")
        elif self.phase == 'testing':
            self.annot_dir = osp.join(self.annot_dir, "test")
        elif self.phase == 'validation':
            self.annot_dir = osp.join(self.annot_dir, "valid")

        label_files = glob(f"{self.annot_dir}/*.txt")

        pairs = list(combinations(label_files, 2))
        for pair in pairs:
            id1 = pair[0].split("/")[-1].split("_")[0]
            id2 = pair[1].split("/")[-1].split("_")[0]
            if id1 == id2:
                self.pos_pairs.append([pair[0], pair[1], 1.0])
            else:
                self.neg_pairs.append([pair[0], pair[1], 0.0])


        # if self.phase == "testing":
        #     self.image_pairs = self.pos_pairs + self.neg_pairs
        # else:
        if data_handling == "balanced":
            N = len(self.pos_pairs)
            if 2 * N < len(self.neg_pairs):
                self.image_pairs = self.pos_pairs + sample(self.neg_pairs, 2 * N)
            else:
                self.image_pairs = self.pos_pairs + self.neg_pairs

        elif data_handling == "randomized":
            print(1 / 0)

        print(f"Phase: {self.phase}, Genuine Pairs: {len(self.pos_pairs)} Imposter Pairs: {len(self.neg_pairs)}")
        print(f"Finally Selected: {len(self.image_pairs)}")

    def __len__(self):
        return len(self.image_pairs)


    def __getitem__(self, index):
        x1 = pil_loader(self.image_pairs[index][0].replace(".txt", ".png"), self.n_channels)
        x2 = pil_loader(self.image_pairs[index][1].replace(".txt", ".png"), self.n_channels)

        if self.crop_box:
            # Reading the bounding box information from label files.
            boxYOLO1 = self.getYOLOLabel(self.image_pairs[index][0])
            boxYOLO2 = self.getYOLOLabel(self.image_pairs[index][1])
            # Cropping out the
            x1 = cropImage(x1, boxYOLO1)
            x2 = cropImage(x2, boxYOLO2)
        else:
            # This step crops the best square from the image.
            x1 = squareCROP(x1)
            x2 = squareCROP(x2)

        # x1 = apply_CLAHE(x1)
        # x2 = apply_CLAHE(x2)

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        y1 = float(self.image_pairs[index][2])

        return x1, x2, y1

    def save_pairs(self):
        pass

    def getYOLOLabel(self, label_path):
        with open(label_path, "r") as fp:
            [cls, x, y, w, h] = list(map(float, fp.readline().split(" ")))
            return [cls, x, y, w, h]


image_size=256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




if __name__ == "__main__":
    ds = SiameseDatasetChextXray14(img_dir="/home/sonymd/Downloads/ChestXray14Data/small_subset",
                                   annot_dir="/home/sonymd/Downloads/ChestXray14Data/small_subset",
                                   phase="training",
                                   crop_box=True,
                                   transform=transform)

    a, b, c = ds[50000]

    print(a.shape, b.shape, c)








