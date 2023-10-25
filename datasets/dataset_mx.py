import io
import time
import mxnet as mx
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt



class MXNetRecDataset(Dataset):
    def __init__(self, rec_path, transform=None):
        self.rec_path = rec_path
        self.idx_path = rec_path[:-4] + ".idx"
        self.lst_path = rec_path[:-4] + ".lst"
        self.transform = transform

        # Open the record file and its corresponding index file
        self.record = mx.recordio.MXIndexedRecordIO(self.idx_path, self.rec_path, 'r')
        self.list_file = open(self.lst_path, "r")
        self.idx2file = self.make_searchable_list()

        # Get the number of items in the record file
        self.length = len(list(self.record.keys))
        print(self.length)

    def __len__(self):
        return self.length

    def make_searchable_list(self):
        with open(self.lst_path, "r") as file:
            lines = file.readlines()
            lines = [line.split("\t") for line in lines]
            idx2file = {line[0]: line[2] for line in lines}

            return idx2file

    def __getitem__(self, idx):
        # Get the record at the given index
        record = self.record.read_idx(idx)
        header, img = mx.recordio.unpack(record)

        # Convert the binary image data to a PIL image
        img = Image.open(io.BytesIO(img))

        # Apply the transformations, if any
        if self.transform:
            img = self.transform(img)

        # Extract the label from the header
        # print(header)
        label = int(header.label)
        filename = self.idx2file[str(idx)]

        return img, label, filename.strip()

def benchmark_loading(dataset):
    start = time.time()
    for i in range(len(dataset)):
        img, label, filename = dataset[i]
        # print(img.shape, label, filename)

    end = time.time()
    print(f"Data Loading Time: {end - start} seconds")


def show_sample_grid(datasets, rows=4, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    for i in range(rows):
        for j in range(cols):
            img, label, filename = datasets[i * cols + j]
            axes[i, j].imshow(img.permute(1, 2, 0))
            axes[i, j].set_title(filename)
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define the transformations you want to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    # Create an instance of the MXNetRecDataset
    ds_train = MXNetRecDataset("../MSU-SPG-Radiography-Dataset/cropped_images/t1-t5/train.rec", transform=transform)
    ds_valid = MXNetRecDataset("../MSU-SPG-Radiography-Dataset/cropped_images/t1-t5/valid.rec", transform=transform)

    # Print the length of the datasets
    print(f"Train Dataset Length: {len(ds_train)}")
    print(f"Valid Dataset Length: {len(ds_valid)}")

    # Benchmark the data loading time
    benchmark_loading(ds_train)


    # Show some samples from the dataset
    show_sample_grid(ds_train)



