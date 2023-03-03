#       inference.py
#       Created by Redwan Sony (sonymd) at 2/5/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU

import argparse
import os
import shutil
from glob import glob

import matplotlib
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import configClass
from dataset import XrayDataset
from train import LightningTrainer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def inference(cfg, best_model_checkpoint):
    ds_valid = XrayDataset(cfg, split="valid", transform=cfg.tfms_valid)
    dl_valid = DataLoader(ds_valid,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          persistent_workers=True)
    print(len(ds_valid))
    model = LightningTrainer.load_from_checkpoint(best_model_checkpoint)
    model_dir = os.path.split(best_model_checkpoint)[0]

    predictions, gt, code_names = [], [], []
    gt = []
    ml_model = model.model.to(f"cuda:{cfg.cuda_devices[0]}")
    ml_model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dl_valid), total=len(dl_valid), desc="Generating Predictions.."):
            X, y, codes = batch
            # print(y.shape)
            y_hat = ml_model(X.to(f"cuda:{cfg.cuda_devices[0]}"))
            preds = model.softmax(y_hat)
            predictions.append(preds.cpu())
            gt.extend(y.tolist())
            code_names.extend(codes)
        all_results = torch.vstack(predictions)
        final_predict = torch.argmax(all_results, axis=1).tolist()

        cm = confusion_matrix(gt, final_predict)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        acc = accuracy_score(gt, final_predict)
        acc_bal = balanced_accuracy_score(gt, final_predict)
        print("Accuracy: ", acc, "Balanced Accuracy:", acc_bal)

        plt.imshow(cm_norm, cmap=plt.cm.gray)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Model: {cfg.model_arch} Box:{cfg.box_preset} Acc:{acc:0.4f}")
        saving_path = f"{model_dir}/{cfg.model_arch}_{cfg.box_preset}_acc_{acc:0.4f}.png"
        plt.savefig(saving_path, dpi=400)
        print("Confusion Matrix Saved To: ", saving_path)

        with open("results/base_experiment/labels_mapping.csv", "r") as fp:
            first = True
            mappings = []
            lines = fp.readlines()
            for line in lines:
                if first:
                    first = False
                else:
                    mappings.append(line.split(",")[0])

        os.makedirs(os.path.join(model_dir, "misclassified_samples"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "correctly_classified_samples"), exist_ok=True)

        with open(f"{model_dir}/{cfg.model_arch}_{cfg.box_preset}_acc_{acc:0.4f}_predictions.csv", "w") as fp:
            print("code_name, ground_truth, prediction", file=fp)
            for code, y_true, y_pred in zip(code_names, gt, final_predict):
                print(f"{code},{mappings[y_true]},{mappings[y_pred]}", file=fp)
                if y_true != y_pred:
                    original_name = f"{code}.png"
                    new_name = f"{code} misclassified as {mappings[y_pred]}.png"

                    shutil.copy(os.path.join(cfg.img_dir, original_name),
                                os.path.join(os.path.join(model_dir, "misclassified_samples"), new_name))
                else:
                    original_name = f"{code}.png"
                    new_name = f"{code} classified as {mappings[y_pred]}.png"

                    shutil.copy(os.path.join(cfg.img_dir, original_name),
                                os.path.join(os.path.join(model_dir, "correctly_classified_samples"), new_name))


def select_the_best_model(save_dir, model_arch, box_preset):
    query = f"{save_dir}/box_{box_preset}/{model_arch}/*.ckpt"
    res = glob(query)
    if len(res):
        accs = [(x.split(".")[-2]) for x in res]
        print(accs)
        max_idx = np.argmax([int(acc) for acc in accs])

        print("Selected Best Model: ", res[max_idx])
        return res[max_idx]
    else:
        print("ERROR: No model found with the given parameters")
        return None


if __name__ == "__main__":
    """ To run this script:
        python train.py --model_name resnet34 --gpu 7 --box_preset 3
    """
    # Setting the configurations
    parser = argparse.ArgumentParser(prog='IFACTS Experiment',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')
    parser.add_argument("--model_arch", type=str, default="resnet34",
                        help="Select the model selection key.\nFor more look for cfr.model_arch in models.py file")
    parser.add_argument("--gpu", type=int, default=7, choices={0, 1, 2, 3, 4, 5, 6, 7},
                        help="Select which gpu you would like to use")
    parser.add_argument("--box_preset", type=int, default=3, choices={0, 1, 2, 3},
                        help="Select the bounding box configuration")
    args = parser.parse_args()
    cfg = configClass()
    cfg.model_arch = args.model_arch
    cfg.cuda_devices = [args.gpu, ]
    cfg.box_preset = args.box_preset

    # Getting the best checkpoint for the configuration
    best_model = select_the_best_model(cfg.checkpoints_dir, cfg.model_arch, cfg.box_preset)
    print(type(best_model))
    if best_model is not None:
        inference(cfg=cfg, best_model_checkpoint=best_model)
