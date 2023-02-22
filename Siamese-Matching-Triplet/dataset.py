import os
import glob
from torch.utils.data import Dataset, DataLoader
import cv2

from torchvision.transforms import Compose, ToTensor, Normalize, Resize


class TripletXRay(Dataset):
    def __init__(self, data_dir, annot_dir, width, height, num_triplets=10000, split="train", transforms=None):
        super().__init__()

        self.data_dir = data_dir
        self.annot_dir = annot_dir
        self.width = width
        self.height = height
        self.num_triplets = num_triplets
        self.split = split
        self.transforms = transforms
        self.all_images = glob.glob(f"{self.data_dir}/{self.split}/*_*.png")
        self.IDs_imgs = {}

        self.IDs = list(set([x.split("/")[-1].split("_")[0] for x in self.all_images]))

        for ID in self.IDs:
            self.IDs_imgs[ID] = []
        for img_path in self.all_images:
            self.IDs_imgs[img_path.split("/")[-1].split("_")[0]].append(img_path)

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        pass


tfms = Compose([ToTensor(),
                Resize(224, 224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == "__main__":
    ds = TripletXRay(data_dir="/home/sonymd/Downloads/ChestXray14Data/small_subset",
                     annot_dir="/home/sonymd/Downloads/ChestXray14Data/small_subset",
                     width=224,
                     height=224,
                     transforms=tfms)
    print(len(ds), ds.IDs_imgs)

    print(len(ds.all_images))
    total = 0

    for ID in ds.IDs:
        members = len(ds.IDs_imgs[ID])
        if members == 1:
            del ds.IDs_imgs[ID]
    print(ds.IDs_imgs)
