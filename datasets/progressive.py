#       datasets/progressive.py
#       Created by Redwan Sony (sonymd) at 11/22/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os.path as osp
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tf
from PIL import Image

random.seed(1455)


def train_test_split(IDs, test_size=0.2, random_state=42):
    random.seed(random_state)
    random.shuffle(IDs)
    test_ids = IDs[:int(len(IDs) * test_size)]
    train_ids = IDs[int(len(IDs) * test_size):]
    return train_ids, test_ids


def print_split_summary(df, mode):
    for split in ['train', 'valid', 'test']:
        df_split = df[df[mode] == split]
        df_split_nih = df_split[df_split['Type of Rad'] == 'NIH']
        df_split_msufal = df_split[df_split['Type of Rad'] == 'Case']
        print(
            f"{split}:\tNIH-CX:{len(df_split_nih['ID'].unique())}\tMSUFAL:{len(df_split_msufal['ID'].unique())}\tTotal:{len(df_split['ID'].unique())}")

    print("\n\n")


def data_split(metadata_file: str, modes: list):
    """

    Args:
        metadata_file: Excel file containing all the information
        modes: one of the modes from ['case_in_test', 'case_in_train', 'case_in_random']
    """
    if not osp.exists(metadata_file):
        raise FileNotFoundError("Metadata file not found at {}".format(metadata_file))
    elif metadata_file.endswith(".csv"):
        df = pd.read_csv(metadata_file)
    elif metadata_file.endswith(".xlsx"):
        df = pd.read_excel(metadata_file, sheet_name='Data input information')
    else:
        raise Exception("Metadata file format not supported")

    # Separate the NIH and MSUFAL Cases
    df_nih = df[df['Type of Rad'] == 'NIH']
    df_msufal = df[df['Type of Rad'] == 'Case']

    # Print brief summary
    print('NIH-CX ID Count:', len(df_nih['ID'].unique()), f"\tRad Count:{len(df_nih)}\n" +
          'MSUFAL ID Count:', len(df_msufal['ID'].unique()), f"\tRad Count:{len(df_msufal)}")

    for mode in modes:
        # Split the NIH-CX into train, val, test
        if mode == 'case_in_test':
            test_ids = df_msufal["ID"].unique()
            train_val_ids = df_nih["ID"].unique()
            train_ids = random.sample(list(train_val_ids), 608)
            val_ids = list(set(train_val_ids) - set(train_ids))
        elif mode == 'case_in_train':
            train_ids = list(df_msufal["ID"].unique()) + list(
                random.sample(list(df_nih["ID"].unique()), 608 - len(df_msufal["ID"].unique())))
            val_test = set(df["ID"].unique()) - set(train_ids)
            val_ids = random.sample(val_test, len(val_test) // 2)
            test_ids = list(val_test - set(val_ids))
        elif mode == 'case_in_random':
            train_ids, val_test_ids = train_test_split(df["ID"].unique(), test_size=0.2, random_state=1455)
            val_ids, test_ids = train_test_split(val_test_ids, test_size=0.5, random_state=1455)

        else:
            raise Exception("Mode not supported")

        # Assign the split to each row
        df[mode] = 'train'
        df.loc[df['ID'].isin(val_ids), mode] = 'valid'
        df.loc[df['ID'].isin(test_ids), mode] = 'test'

        # Print brief summary
        print("Split Mode:", mode)
        print_split_summary(df, mode)

    # Save the split
    df.to_excel(metadata_file.replace('.xlsx', f'_splitted.xlsx'), index=False, sheet_name='Data input information')


class ProgressiveBaseDataset(Dataset):
    def __init__(self, images_dir, mtdt_file, split_mode, box_preset, split, transform=None):
        """
        Args:
            mtdt_file: metadata file path containing the split information
            split_mode: one of the modes from ['case_in_test', 'case_in_train', 'case_in_random']
            split: train, valid or test
            transform: transform to be applied on the images
            box_preset: one of the bounding box presets from ['t1-t5', 'complete-vertebrae', clavicle-only', 'whole']
        """
        if not osp.exists(mtdt_file):
            raise FileNotFoundError("Metadata file not found at {}".format(mtdt_file))
        elif mtdt_file.endswith(".csv"):
            self.df = pd.read_csv(mtdt_file)
        elif mtdt_file.endswith(".xlsx"):
            self.df = pd.read_excel(mtdt_file, sheet_name='Data input information')
        self.images_dir = images_dir
        self.df = self.df[self.df[split_mode] == split].reset_index(drop=True)
        self.transform = transform
        self.box_preset = box_preset
        self.img_dir = osp.join(self.images_dir)

        # Add new column to the dataframe with the image path
        self.df['img_path'] = self.df.apply(lambda row: osp.join(self.img_dir, row['IFACTS File Name'] + '.png'),
                                            axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pass


class ProgressiveTrainDataset(ProgressiveBaseDataset):
    def __init__(self, images_dir, mtdt_file, split_mode, box_preset, split, transform=None):
        super().__init__(images_dir, mtdt_file, split_mode, box_preset, split, transform)

        self.classes = sorted(self.df['ID'].unique())

        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2class = {idx: cls for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row of the dataframe
        file_info = self.df.iloc[idx]
        file_name = file_info['IFACTS File Name']
        img = Image.open(file_info['img_path']).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.class2idx[file_info['ID']]
        return img, label #, file_name, dict(file_info)


class ProgressiveValidDataset(ProgressiveBaseDataset):
    def __init__(self, images_dir, mtdt_file, split_mode, box_preset, split, transform=None):
        super().__init__(images_dir, mtdt_file, split_mode, box_preset, split, transform)

        self.classes = sorted(self.df['ID'].unique())

        self.class2idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx2class = {idx: cls for idx, cls in enumerate(self.classes)}


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row of the dataframe
        file_info = self.df.iloc[idx]
        file_name = file_info['IFACTS File Name']
        img = Image.open(file_info['img_path']).convert('RGB')

        if self.transform:
            img = self.transform(img)
        label = self.class2idx[file_info['ID']]

        return img, label #  file_name, dict(file_info)


class ProgressiveTestDataset(ProgressiveBaseDataset):
    def __init__(self, images_dir, mtdt_file, split_mode, box_preset, split, transform=None):
        super().__init__(images_dir, mtdt_file, split_mode, box_preset, split, transform)

    def __len__(self):
        1

    def __getitem__(self, item):
        return 1, 2


def get_progressive_datasets(cfg):
    # Load the datasets
    ds_train = ProgressiveTrainDataset(images_dir=cfg.dataset_dir,
                                       mtdt_file=cfg.metadata_file,
                                       split_mode=cfg.split_mode,
                                       box_preset=cfg.box_preset,
                                       split='train',
                                       transform=cfg.tfms_train)
    ds_valid = ProgressiveTrainDataset(images_dir=cfg.dataset_dir,
                                       mtdt_file=cfg.metadata_file,
                                       split_mode=cfg.split_mode,
                                       box_preset=cfg.box_preset,
                                       split='valid',
                                       transform=cfg.tfms_valid)
    ds_test = ProgressiveTrainDataset(images_dir=cfg.dataset_dir,
                                      mtdt_file=cfg.metadata_file,
                                      split_mode=cfg.split_mode,
                                      box_preset=cfg.box_preset,
                                      split='test',
                                      transform=cfg.tfms_valid)

    # Load the dataloaders
    dl_train = DataLoader(ds_train,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers,
                          pin_memory=True)
    dl_valid = DataLoader(ds_valid,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers,
                          pin_memory=True)
    dl_test = DataLoader(ds_test,
                         batch_size=cfg.batch_size,
                         shuffle=False,
                         num_workers=cfg.num_workers,
                         pin_memory=True)

    return ds_train, ds_valid, ds_test, dl_train, dl_valid, dl_test


if __name__ == "__main__":
    # # Testing for the data_split function
    # dataset_dir = "/research/iprobe-sonymd/MSU-SPG-Radiography-Dataset"
    # images_dir = osp.join(dataset_dir, "cropped-combined")
    # metadata_file = osp.join(dataset_dir, "IFACTS MASTER_github.xlsx")
    # split_modes = ['case_in_test', 'case_in_train', 'case_in_random']
    #
    # if os.path.exists(metadata_file.replace('.xlsx', f'_splitted.xlsx')):
    #     print("Split already done")
    # else:
    #     data_split(metadata_file=metadata_file, modes=split_modes)
    #     print("Split done")

    box_preset = 't1-t5'
    split_mode = 'case_in_test'

    # ImageNet statistics as default
    stat_mean = [0.485, 0.456, 0.406]
    stat_std = [0.229, 0.224, 0.225]

    tfms_train = tf.Compose([
        tf.ToTensor(),
        tf.Resize((512, 512)),
        tf.RandomRotation(degrees=15),
        tf.RandomPerspective(distortion_scale=0.2, p=0.3),
        tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
        tf.Normalize(mean=stat_mean, std=stat_std)
    ])

    tfms_valid = tf.Compose([
        tf.ToTensor(),
        tf.Resize((512, 512)),
        tf.Normalize(mean=stat_mean, std=stat_std)
    ])

    # Test Loading the dataset
    ds = ProgressiveTrainDataset(images_dir="/research/iprobe-sonymd/MSU-SPG-Radiography-Dataset/cropped-combined",
                                 mtdt_file="/research/iprobe-sonymd/MSU-SPG-Radiography-Dataset/IFACTS MASTER_github_splitted.xlsx",
                                 split_mode='case_in_test',
                                 box_preset='t1-t5',
                                 split='test',
                                 transform=tfms_train)
    print(len(ds))

    print(ds[0]["img_path"])
