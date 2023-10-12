# ensemble.py
# Created by sonymd at 2/24/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import argparse
import torch
from torch import nn
import numpy as np
from itertools import permutations
from train import LightningTrainer
from dataset import XrayDataset
from torch.utils.data import DataLoader
from configs.config import configClass
from inference import select_the_best_model
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score


def get_model_permutations(num_models):
    idxs = list(range(num_models))
    all_combs = []
    for i in range(len(idxs)+1):
        perm = permutations(idxs, i)
        for j in list(perm): 
            all_combs.append(tuple(sorted(list(j))))
    return list(set(all_combs))

def get_predictions(save_dir, model_arch, box_preset, data_loader, device=torch.device("cpu")):
    # Loading the model
    model_ckpt_path = select_the_best_model(save_dir=save_dir, model_arch=model_arch, box_preset=box_preset)
    model_dir = os.path.split(model_ckpt_path)[0]
    model = LightningTrainer.load_from_checkpoint(model_ckpt_path).model.to(device=device).eval()
    softmax = nn.Softmax(dim=1)
    # Generating the predictions
    _y_true, _y_pred, _codes = [], [], [],
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            X, y, code = batch
            logits = model(X.to(device))
            probs = softmax(logits)
            _y_pred.append(probs.cpu())
            _y_true.extend(y)
            _codes.extend(code)
    _predict_probs = torch.vstack(_y_pred)
    _y_pred = torch.argmax(_predict_probs, axis=1).tolist()

    torch.cuda.empty_cache()

    return _y_true, _y_pred, _codes, _predict_probs.numpy()


def predict_models(cfg, model_archs, box_presets):
    pass


def prediction_sanity_check(_y_true, _y_pred, _codes, _predict_probs):
    print("y_true length:", len(_y_true), type(_y_true))
    print("y_pred length:", len(_y_pred), type(_y_pred))
    print("y_code length:", len(_codes), type(_codes))
    print("Prediction Probability Matrix: ", _predict_probs.shape, type(_predict_probs))
    # if len(_y_true) == len(_y_pred) == len(_codes) == len(_predict_probs):
    #     print("valid prediction")
    #     ret
    # else:
    #     print("Check your predictions again")


def ensemble(base_config: configClass, configs: list, mode: str = "mean"):
    """
    :param base_config: This is the modified base configuration to make the dataloader work flawlessly
    :param configs: This is the list of model configurations that we need to ensemble
    :param mode: This is the mode of the ensemble. The default one 'mean' indicates an average ensemble.
    :return:
    """
    all_predictions = {}
    for _save_dir, _model_arch, _box_preset in configs:
        # Resetting the experiment configuration to load the data of appropriate configurations.
        base_config.model_arch = _model_arch
        base_config.box_preset = _box_preset
        base_config.checkpoints_dir = _save_dir

        # Loading the Data
        _ds_valid = XrayDataset(cfg, split="valid", transform=base_config.tfms_valid)
        _dl_valid = DataLoader(_ds_valid,
                               batch_size=base_config.batch_size,
                               shuffle=False,
                               num_workers=base_config.num_workers,
                               pin_memory=base_config.pin_memory,
                               persistent_workers=True)

        # Getting Prediction for a single model
        _y_true, _y_pred, _codes, _predict_probs = get_predictions(save_dir=_save_dir,
                                                                   model_arch=_model_arch,
                                                                   box_preset=_box_preset,
                                                                   data_loader=_dl_valid,
                                                                   device=base_config.cuda_devices[0])

        prediction_sanity_check(_y_true, _y_pred, _codes, _predict_probs)
        all_predictions[(_save_dir, _model_arch, _box_preset)] = [_y_true, _y_pred, _codes, _predict_probs]


    combs = get_model_permutations(len(configs))
    
    for comb in combs:
        pred_probs = np.zeros_like(_predict_probs)
        comb = list(comb)
        if len(comb) > 1:
            for model_idx in comb:
                _save_dir, _model_arch, _box_preset = configs[model_idx]
                _y_true, _y_pred, _codes, _predict_probs = all_predictions[(_save_dir, _model_arch, _box_preset)] 

                pred_probs = pred_probs+_predict_probs
                print((_save_dir, _model_arch, _box_preset), end=" + ")
        
        
            ensemble_pred_probs = pred_probs/len(comb)
            final_predict = torch.argmax(torch.tensor(ensemble_pred_probs), axis=1).tolist()
            acc = accuracy_score(y_true=_y_true, y_pred = final_predict)
            print(":", acc)
                



def ensemble_predictions():
    pass


if __name__ == "__main__":
    # Setting the configurations
    parser = argparse.ArgumentParser(prog='IFACTS Experiment',
                                     description='What the program does',
                                     epilog='Text at the bottom of help')

    parser.add_argument("--gpu", type=int, default=7, choices={0, 1, 2, 3, 4, 5, 6, 7},
                        help="Select which gpu you would like to use")

    args = parser.parse_args()
    cfg = configClass()
    cfg.cuda_devices = [args.gpu, ]


    configs = [["saved_img_img_img", "resnet50", 0],
               ["saved_img_img_img", "resnet50", 1],
               ["saved_img_img_img", "resnet50", 2],
               ["saved_img_img_img", "resnet50", 3]]

    ensemble(base_config=cfg,
             configs=configs,
             mode="mean")

