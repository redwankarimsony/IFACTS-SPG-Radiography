#       datasets/verification.py
#       Created by Redwan Sony (sonymd) at 10/24/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import io
import mxnet as mx
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tf
from PIL import Image
from tqdm import tqdm
import random
from glob import glob





# class ChextXrayVerification(Dataset):
#     def __init__(self, rec_path, transform=None):
#         self.rec_path = rec_path
#         self.idx_path = rec_path[:-4] + ".idx"
#         self.lst_path = rec_path[:-4] + ".lst"
#         self.transform = transform

#         # Open the record file and its corresponding index file
#         self.record = mx.recordio.MXIndexedRecordIO(self.idx_path, self.rec_path, 'r')
#         self.list_file = open(self.lst_path, "r")
#         self.idx2file = self.make_searchable_list()

#         # Get the number of items in the record file
#         self.length = len(list(self.record.keys))

#         # Get the idx to original ID mapping
#         self.label2id = self.label2ID()
#         print("Total Radiographs:", self.length)

#         self.pairs = self.make_pairs()

#     def __len__(self):
#         return len(self.pairs)
    
#     def make_searchable_list(self):
#         with open(self.lst_path, "r") as file:
#             lines = file.readlines()
#             lines = [line.split("\t") for line in lines]
#             idx2file = {line[0]: line[2] for line in lines}

#             return idx2file
        
#     def label2ID(self):
#         label2id = {}
#         for idx in range(self.length):
#             record = self.record.read_idx(idx)
#             header, img = mx.recordio.unpack(record)
#             label2id[header.label] = self.idx2file[str(idx)].split("/")[0]
#         return label2id
    

#     def make_pairs(self):
#         positive_pairs = []
#         negative_pairs = []

#         for idx1 in tqdm(range(self.length), desc="Making Pairs"):
#             for idx2 in range(idx1, self.length):
#                 if idx1 == idx2:
#                     continue
#                 else:
#                     record1 = self.record.read_idx(idx1)
#                     record2 = self.record.read_idx(idx2)
#                     header1, img1 = mx.recordio.unpack(record1)
#                     header2, img2 = mx.recordio.unpack(record2)

#                     if header1.label == header2.label:
#                         positive_pairs.append((img1, header1, img2, header2, 1))
#                     else:
#                         negative_pairs.append((img1, header1, img2, header2, 0))

#         random.shuffle(negative_pairs)
#         negative_pairs = negative_pairs[:len(positive_pairs)]

#         print("Total Positive Pairs:", len(positive_pairs))
#         print("Total Negative Pairs:", len(negative_pairs))

#         return positive_pairs+negative_pairs
    


#     def __getitem__(self, idx):
#         img1, header1, img2, header2, verification_label = self.pairs[idx]
#         img1 = Image.open(io.BytesIO(img1))
#         img2 = Image.open(io.BytesIO(img2))

#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
        
#         label1 = int(header1.label)
#         label2 = int(header2.label)

            
#         filename1 = self.idx2file[str(idx)]
#         filename2 = self.idx2file[str(idx)]

#         return img1, label1, filename1.strip(), img2, label2, filename2.strip(), verification_label
    

class ChextXrayVerificationDataset(Dataset):
    def __init__(self, data_dir, split="valid", transform=None):
        self.dir_path = os.path.join(data_dir, split)
        print(os.path.exists(self.dir_path))
        self.transform = transform

        self.image_paths =  glob(os.path.join(self.dir_path, "*/*.png"))
        print(len(self.image_paths))


        self.pairs = self.make_pairs()

    def __len__(self):
        return len(self.pairs)


    def make_pairs(self):
        positive_pairs, negative_pairs = [], []
        for idx1, img_path1 in tqdm(enumerate(self.image_paths), desc="Making Pairs"):
            for idx2, img_path2 in enumerate(self.image_paths[idx1:]):
                if img_path1 == img_path2:
                    continue
                else:
                    label1 = img_path1.split("/")[-2]
                    label2 = img_path2.split("/")[-2]

                    if label1 == label2:
                        positive_pairs.append((img_path1, img_path2, 1))
                    else:
                        negative_pairs.append((img_path1, img_path2, 0))
    
        print("Total Positive Pairs:", len(positive_pairs))
        print("Total Negative Pairs:", len(negative_pairs))
        random.shuffle(negative_pairs)

        return positive_pairs + negative_pairs[:len(positive_pairs)]
    

    def __getitem__(self, index):
        img_path1, img_path2, verification_label = self.pairs[index]
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, verification_label, img_path1, img_path2
    






if __name__=="__main__":
    tfms_valid= tf.Compose([
            tf.ToTensor(),
            tf.Resize((512, 512)),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    


    ds = ChextXrayVerificationDataset(rec_path="/localscratch2/sonymd/MSU-SPG-Radiography-Dataset/cropped_images/t1-t5/valid.rec", 
                               transform=tfms_valid)
    

    for x in ds:
        img1, img2, verification_label, img_path1, img_path2 = x

        print(img1.shape, img2.shape, img_path1.split("/")[-1], img_path2.split("/")[-1], verification_label)
    





    

    