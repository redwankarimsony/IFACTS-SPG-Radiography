import os
import cv2
import torch
import numpy as np

from matplotlib import patches

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader, Subset
import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class ObjectDetectionDataset(Dataset):
    """
    make sure you have all the images in the same folder and the
    label file is pascal voc xml format for each of the image.
    """

    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width

        # sorting the images for consistency
        # To get images, the extension of the filename is checked to be jpg
        self.imgs = self.list_image_files(self.files_dir)

        # classes: 0 index is reserved for background
        self.classes = ["background", 'bone']

    def __getitem__(self, idx):

        image_path = self.imgs[idx]
        # image_path = os.path.join(self.files_dir, img_name)

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        # print(img.shape)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        # annotation file
        annot_file_path = image_path[:-4] + '.xml'

        boxes, labels = [], []
        tree = et.parse(annot_file_path)
        root = tree.getroot()

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))

            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)

            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)

            xmin_corr = (xmin / wt) * self.width
            xmax_corr = (xmax / wt) * self.width
            ymin_corr = (ymin / ht) * self.height
            ymax_corr = (ymax / ht) * self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=target['boxes'],
                                     labels=labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def list_image_files(img_dir, exts=["JPG", "PNG", "BMP", "JPEG"]):
        all_files = os.listdir(img_dir)
        exts = exts + [x.lower() for x in exts]
        img_files = []
        for file_path in all_files:
            ext = file_path.split(".")[-1]
            if ext in exts:
                img_files.append(os.path.join(img_dir, file_path))
        return img_files

    @staticmethod
    def plot_img_bbox(img, target):
        # plot the image and bboxes
        # Bounding boxes are defined as follows: x-min y-min width height
        fig, a = plt.subplots(1, 1)
        fig.set_size_inches(5, 5)
        a.imshow(img)
        for box in (target['boxes']):
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y),
                                     width, height,
                                     linewidth=2,
                                     edgecolor='r',
                                     facecolor='none')

            # Draw the bounding box on top of the image
            a.add_patch(rect)
        plt.savefig("labeled_plot.png", dpi=300)


# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def load_dataset(config):
    # use our dataset and defined transformations
    dataset = ObjectDetectionDataset(config["train_dir"],
                                     config["width"],
                                     config["height"],
                                     transforms=get_transform(train=True))
    dataset_test = ObjectDetectionDataset(config["test_dir"],
                                          config["width"],
                                          config["height"],
                                          transforms=get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset) * test_split)
    dataset = Subset(dataset, indices[:-tsize])
    dataset_valid = Subset(dataset_test, indices[-tsize:])

    # define training and validation data loaders
    dl_train = DataLoader(dataset,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          num_workers=config["num_workers"],
                          pin_memory=config["pin_memory"],
                          collate_fn=utils.collate_fn)

    dl_valid = DataLoader(dataset_valid,
                          batch_size=config["batch_size"],
                          shuffle=True,
                          num_workers=config["num_workers"],
                          pin_memory=config["pin_memory"],
                          collate_fn=utils.collate_fn)

    dl_test = DataLoader(dataset_test,
                         batch_size=config["batch_size"] * 2,
                         shuffle=False,
                         num_workers=config["num_workers"],
                         pin_memory=config["pin_memory"],
                         collate_fn=utils.collate_fn)

    return dl_train, dl_valid, dl_test


if __name__ == "__main__":
    # defining the files directory and testing directory
    files_dir = './dummy_dataset'

    # check dataset
    dataset = ObjectDetectionDataset(files_dir, 224, 224)
    print('length of dataset = ', len(dataset), '\n')

    # # getting the image and target for a test index.  Feel free to change the index.
    img, target = dataset[78]
    print(img.shape, '\n', target)

    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.savefig("output.png")

    # Function to visualize bounding boxes in the image

    # plotting the image with bboxes. Feel free to change the index
    img, target = dataset[25]
    # plot_img_bbox(img, target)
