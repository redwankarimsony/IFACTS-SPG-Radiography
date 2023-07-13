import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageQualityDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_files, self.labels = self._get_image_files_and_labels()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[index]
        return image, label

    def _get_image_files_and_labels(self):
        image_files = []
        labels = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    image_files.append(os.path.join(root, file))
                    # Assuming label is extracted from the image filename or directory structure
                    label = self._extract_label(file)
                    labels.append(label)
        return image_files, labels

    def _extract_label(self, image_filename):
        # Implement your logic to extract the label from the image filename or directory structure
        # Considering the filename looks like this 1360_PM_Chest_AP3_e_1.png and 1 being the label
        
        label = int(image_filename.split(".")[0].split("_")[-1])  # Extract the label based on your dataset's structure
        return label

# Usage example:

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    data_path = '/path/to/dataset/folder'
    
    dataset = ImageQualityDataset(data_path=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
