import os.path as osp
from torch.utils import data
import numpy as np
from PIL import Image
from itertools import combinations
from glob import glob
from random import sample

import torchvision.transforms as transforms
from enhancement_experiments import applyCLAHE


class SiameseDatasetIFACTS(data.Dataset):
    """
    This is the data loader for the old training examples. Where the whole xray image is taken. 
    For genuine pair, label is 1 and for imposter image, label is 0
    """

    def __init__(self, img_dir, annot_dir, phase='testing', data_handling="balanced", n_channels=3, n_samples=32131,
                 transform=None, crop_box=True):
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
        self.image_pairs_path = osp.join(".", "data_pairs.txt")

        if self.phase == 'training':
            self.annot_dir = osp.join(self.annot_dir, "train")
        elif self.phase == 'testing':
            self.annot_dir = osp.join(self.annot_dir, "test")
        elif self.phase == 'validation':
            self.annot_dir = osp.join(self.annot_dir, "valid")

        label_files = glob(f"{self.annot_dir}/*.txt")
        filenames = [x.split("/")[-1].split(".")[0] for x in label_files]
        imgFiles = []
        for filename in filenames:
            ID = filename.split("_")[0]
            imgFiles.append(glob(osp.join(self.img_dir, ID, f"{filename}.*"))[0])

        pairs = list(combinations(imgFiles, 2))
        for pair in pairs:
            id1 = pair[0].split("/")[-1].split("_")[0]
            id2 = pair[1].split("/")[-1].split("_")[0]
            if (id1 == id2):
                self.pos_pairs.append([pair[0], pair[1], 1.0])
            else:
                self.neg_pairs.append([pair[0], pair[1], 0.0])

        if self.phase == "testing":
            self.image_pairs = self.pos_pairs + self.neg_pairs
        else:
            if data_handling == "balanced":
                N = len(self.pos_pairs)
                if 2 * N < len(self.neg_pairs):
                    self.image_pairs = self.pos_pairs + sample(self.neg_pairs, 2 * N)
                else:
                    self.image_pairs = self.pos_pairs + self.neg_pairs

            elif data_handling == "randomized":
                print(1 / 0)

        print(f"Phase: {self.phase}, Genuine Pairs: {len(self.pos_pairs)} Imposter Pairs: {len(self.neg_pairs)}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        x1 = pil_loader(self.image_pairs[index][0], self.n_channels)
        x2 = pil_loader(self.image_pairs[index][1], self.n_channels)

        if self.crop_box:
            # Reading the bounding box information from label files.
            boxYOLO1 = self.getLabel(self.image_pairs[index][0])
            boxYOLO2 = self.getLabel(self.image_pairs[index][1])
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

    def save_pairs():
        pass

    def getLabel(self, img_path):
        filename = img_path.split("/")[-1].split(".")[0]
        with open(osp.join(self.annot_dir, f"{filename}.txt"), "r") as fp:
            [cls, x, y, w, h] = list(map(float, fp.readline().split(" ")))
            return [cls, x, y, w, h]


def squareCROP(img):
    """
    This function takes an image and return the center cropped of that image.
    """
    W, H = img.size
    if W == H:
        return img
    else:
        extra_pixels = abs(H - W) // 2
        if H > W:
            print("H>W")
            img = img.crop((0, extra_pixels, W, H-extra_pixels))
        else:
            print("H<W")
            img = img.crop((extra_pixels, 0, W-extra_pixels, H))
        return img



def cropImage(img, boxYOLO):
    W, H = img.size
    xmin, ymin, xmax, ymax = bbox2points(boxYOLO[1:], H, W)
    return img.crop((xmin, ymin, xmax, ymax))


def bbox2points(bbox, H, W):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round((x - (h / 2)) * W))
    xmax = int(round((x + (h / 2)) * W))
    ymin = int(round((y - (h / 2)) * H))
    ymax = int(round((y + (h / 2)) * H))
    return xmin, ymin, xmax, ymax


def pil_loader(path, n_channels):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if n_channels == 1:
            return img.convert('L')
        elif n_channels == 3:
            return img.convert('RGB')
        else:
            raise ValueError('Invalid value for parameter n_channels!')


def apply_CLAHE(img):
    img = np.asarray(img)
    return Image.fromarray(applyCLAHE(img, display=False))


image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    ds = SiameseDatasetIFACTS(img_dir="/home/sonymd/Downloads/Chest Radiograph Files/Chest_Radiograph",
                              annot_dir="/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/annotation_T1-T5",
                              phase='testing',
                              transform=transform)
