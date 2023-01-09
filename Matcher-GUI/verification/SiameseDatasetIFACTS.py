from torch.utils import data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class SiameseDatasetIFACTS(data.Dataset):
    def __init__(self, phase='testing', n_channels=3, n_samples=32131, transform=None, 
                image_path='/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_1/',
                image_pairs = None, save_path=None):

        self.phase = phase
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.transform = transform
        self.PATH = image_path
        self.image_pairs = image_pairs



    def __len__(self):
        return len(self.image_pairs)


    def __getitem__(self, index):

        x1 = pil_loader(self.PATH + self.image_pairs[index][0], self.n_channels)
        x2 = pil_loader(self.PATH + self.image_pairs[index][1], self.n_channels)

        id1 = self.image_pairs[index][0].split("_")[0]
        id2 = self.image_pairs[index][1].split("_")[0]

        if self.transform is not None:
            x1 = self.transform(x1)
            x2 = self.transform(x2)
        if id1 == id2:
            y1 = float(1.0)
        else:
            y1 = float(0.0)

        return x1, x2, y1


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

image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



if __name__ == "__main__":
    ds = SiameseDatasetIFACTS(image_path='/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_1/',
                                transform=transform)
    print(len(ds))
    a, b, c = ds[0]
    print(a.shape, b.shape, c)
