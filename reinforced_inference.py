import os.path as osp
import torch
import pickle

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from dataset import XrayDataset
from train import LightningTrainer
from inference import select_the_best_model
from configs.config import configClass
from subprocess import run



def inference_on_split(cfg, split="valid"):
    ds_train = XrayDataset(cfg, split=split, transform=cfg.tfms_valid)
    dl_train = DataLoader(ds_train,
                          batch_size=cfg.batch_size,
                          shuffle=False,
                          num_workers=cfg.num_workers,
                          pin_memory=cfg.pin_memory,
                          persistent_workers=True)
    print(len(ds_train))
    best_model_checkpoint = select_the_best_model(cfg.checkpoints_dir, 
                              model_arch=cfg.model_arch, 
                              box_preset=cfg.box_preset)
    
    model = LightningTrainer.load_from_checkpoint(best_model_checkpoint)
    model_dir = osp.split(best_model_checkpoint)[0]

    predictions, gt, code_names = [], [], []
    ml_model = model.model.to(f"cuda:{cfg.cuda_devices[0]}")
    ml_model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Generating Predictions on {split}.."):
            X, y, codes = batch
            # print(y.shape)
            y_hat = ml_model(X.to(f"cuda:{cfg.cuda_devices[0]}"))
            preds = model.softmax(y_hat)
            predictions.append(preds.cpu())
            gt.extend(y.tolist())
            code_names.extend(codes)
        all_results = torch.vstack(predictions)
        final_predict = torch.argmax(all_results[:, :679], axis=1).tolist()

        cm = confusion_matrix(gt, final_predict)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        acc = accuracy_score(gt, final_predict)
        acc_bal = balanced_accuracy_score(gt, final_predict)
        print("Accuracy: ", acc, "Balanced Accuracy:", acc_bal)
    
    # saving the predictions
    results = {code:vec for code, vec in zip(code_names, all_results)}

    with open(osp.join(model_dir, f"predictions_on_{split}.pkl"), "wb") as fp:
        pickle.dump(results, fp)

    return all_results, code_names







def generate_train_prediction_on_all(checkpoints_dir:str,  models = [], box_presets = []):
    for model in models:
        for box_preset in box_presets:
            cfg = configClass()
            cfg.checkpoints_dir = checkpoints_dir
            cfg.model_arch = model
            cfg.box_preset=box_preset

            # run("ulimit -n 2048")
            inference_on_split(cfg, split="valid")
            inference_on_split(cfg, split="train")
            print(model, box_preset, "Calculated")


def reinforcement_inference(checkpoints_dir, model_arch, box_preset):
    best_model = select_the_best_model(save_dir=checkpoints_dir, model_arch=model_arch, box_preset=box_preset)
    model_dir = osp.split(best_model)[0]

    with open(osp.join(checkpoints_dir, f"box_{box_preset}", model_arch, "predictions_on_train.pkl"), "rb") as fp:
        train_predicts = pickle.load(fp)

    with open(osp.join(checkpoints_dir, f"box_{box_preset}", model_arch, "predictions_on_valid.pkl"), "rb") as fp:
        valid_predicts = pickle.load(fp)

    print(valid_predicts.keys())

    








if __name__ == "__main__":
    checkpoints_dir = "saved_iii"
    box_presets = [0,1,2,3]
    models = ["resnet34", "resnet50", "resnet101", "densenet121", "densenet161", "densenet169", "densenet201", 
              "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]

    # generate_train_prediction_on_all(checkpoints_dir=checkpoints_dir,
    #                                  models=models,
    #                                  box_presets=box_presets)

    reinforcement_inference(
        checkpoints_dir="saved_iii",
        model_arch="resnet34",
        box_preset = 0
    )

    
    
    




