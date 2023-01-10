import os

import numpy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from .networks.SiameseNetwork import SiameseNetwork
from .SiameseDatasetIFACTS import SiameseDatasetIFACTS

# from utils import Utils
# from sklearn import metrics
# import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import numpy as np
# import argparse
from tqdm import tqdm

image_size = 256

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def loadModel():
    # Load the pre-trained model
    savedModel = os.path.join(os.path.dirname(__file__),
                              'trained_models/verification_approach_final_model_IFACTS.pth')
    if torch.cuda.is_available():
        model = SiameseNetwork(network='ResNet-50', in_channels=3, n_features=128).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(savedModel))
    else:
        model = SiameseNetwork(network='ResNet-50', in_channels=3, n_features=128)
        model.load_state_dict(
            torch.load(savedModel, map_location='cpu'))

    return model


def getMatchScores(model, probe, gallery, imgDir):
    pairs = [(probe, x) for x in gallery]

    ds_test = SiameseDatasetIFACTS(transform=transform, image_path=imgDir, image_pairs=pairs)
    dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=12, pin_memory=True)

    # Testing phase
    model.eval()
    y_true = None
    y_pred = None

    print('Testing----->')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dl_test)):
            inputs1, inputs2, labels = batch

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            if torch.cuda.is_available():
                inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            outputs = model(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)

            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

            # print('Progress: ' + str(np.round((i + 1) * 100 / len(test_loader), 2)) + '%')

    y_pred = y_pred.squeeze()

    return gallery, y_true, y_pred


def getTopKPredictions(probe: str, galleryDir: str, K=10):
    allFiles, imgFiles = os.listdir(galleryDir), []
    for fileName in allFiles:
        if not fileName.endswith("txt"):
            imgFiles.append(fileName)
    imgFiles = sorted(imgFiles)
    if probe in imgFiles:
        imgFiles.remove(probe)

    imageNames, y_true, y_pred = getMatchScores(loadModel(), probe=probe, gallery=imgFiles, imgDir=galleryDir)

    results = [[a, b.item(), c.item()] for a, b, c in zip(imageNames, y_true, y_pred)]

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    # adding image paths
    for i in range(len(sorted_results)):
        sorted_results[i][0] = os.path.join(galleryDir, sorted_results[i][0])

    return sorted_results[:K]


if __name__ == "__main__":
    imgDir = "/home/sonymd/Desktop/IFACTS/IFACTS-SPG-Radiography/Annotation/images_annotation_1/"
    allFiles, imgFiles = os.listdir(imgDir), []
    for fileName in allFiles:
        if not fileName.endswith("txt"):
            imgFiles.append(fileName)
    imgFiles = sorted(imgFiles)

    probe = imgFiles[108]
    imgFiles.remove(probe)

    imageNames, y_true, y_pred = getMatchScores(loadModel(), probe=probe, gallery=imgFiles, imgDir=imgDir)

    results = [(a, b.item(), c.item()) for a, b, c in zip(imageNames, y_true, y_pred)]

    # def score(x):
    #     return x[2]

    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)
    print(probe)
    for res in sorted_results:
        print(res)
