#       inference.py
#       Created by Redwan Sony (sonymd) at 10/19/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import argparse
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import mxnet as mx
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt



from train_base import LightningTrainer, get_datasets
from configs.config import ConfigClass
from datasets import MXNetRecDataset
from utils import get_gpu_with_least_memory_over_period
from summarize import get_the_best_model


torch.set_float32_matmul_precision("high")

__all__ = ["get_top_k_predictions", 
           "get_top_k_accuracies", 
           "inference", 
           "get_model_summary"]



def get_top_k_predictions(y_hat, k=3):
    """ Given the softmax output, get the predicted labels and top-k predictions

    Args:
        y_hat (_type_): The softmax output of the model
        k (int, optional): Order. Defaults to 5.

    Returns:
        top_k_labels (): top-k labels
        top_k_probs (): top-k probabilities
    """
    # Get the top-k predictions
    top_k = torch.topk(y_hat, k=k, dim=1)
    # Get the top-k labels
    top_k_labels = top_k.indices
    # Get the top-k probabilities
    top_k_probs = top_k.values

    return top_k_labels, top_k_probs


def get_top_k_accuracies(y_hats, y_trues, k_max=5):
    """_summary_

    Args:
        y_hats (_type_): The softmax output of the model
        y_trues (_type_): The true labels of the images
        k_max (int, optional): The order upto which accuracy should be calculated. Defaults to 5.

    Returns:
        _type_: top-k accuracies for each order
    """

    accs = []

    for k in range(1, k_max+1):
        # Get top-k predictions
        top_k_labels, top_k_probs = get_top_k_predictions(y_hats, k=k)

        top_k_predictions = []
        for y_true, top_k_label in zip(y_trues, top_k_labels):
            if y_true.item() in top_k_label.tolist():
                top_k_predictions.append(1.)
            else:
                top_k_predictions.append(0.)

        acc = torch.mean(torch.tensor(top_k_predictions)).item()
        accs.append(acc)

    return accs
    




def inference(saved_model_path):
    """Takes the pytorch lightning path as the input path and returns the predictions

    Args:
        saved_model_path (_type_): relative path of the saved model with extension .ckpt

    Returns:
        y_hats (torch.ndarray): Softmax output of the model of shape (N, num_classes)
        y_trues (list): List of true labels of the images
        filenames (list): List of filenames of the images
    """

    # Load the model
    classifier = LightningTrainer.load_from_checkpoint(saved_model_path)
    classifier.eval()
    print("Loaded Model Successfully")

    # Load the datasets
    ds_train, ds_valid, dl_train, dl_valid = get_datasets(classifier.cfg)
    print("Loaded Datasets Successfully")

    # Send the model to the GPU
    classifier.cuda()


    
    with torch.no_grad():
        y_hats, y_trues, all_filenames = [], [], []

        # Iterate Over the validation dataloader
        for i, batch in tqdm(enumerate(dl_valid), total=len(dl_valid), desc="Inference"):
            images, labels, filenames = batch

            # Forward pass
            y_hat = classifier.model(images.cuda())
            y_hat = F.softmax(y_hat, dim=1)

            # Append the results
            y_hats.append(y_hat.cpu())
            y_trues.extend(labels.cpu().tolist())
            all_filenames.extend(filenames)

    # Convert the list of tensors to a single tensor
    y_hats = torch.concatenate(y_hats, dim=0)
    y_trues = torch.tensor(y_trues)
    
    return y_hats, y_trues, all_filenames




def get_model_summary(experiment_name, model_arch, box_preset, top_ks, gpu):
    """_summary_

    Args:
        experiment_name (_type_): Name of the experiment
        model_arch (_type_): one of the model architectures from the config.py file
        box_preset (_type_): one of the box presets from the config.py file
        top_ks (_type_): upto what order to calculate the accuracy
        gpu (_type_): the gpu to use
        return_predictions (bool, optional): if true returns the logits also.

    Returns:
        accs: a list of accuracies for each order in top_ks
        y_hats: the logits of the model output
    """
    
    # Get the best saved model path from the experiment name, box preset and model architecture
    saved_model_path = get_the_best_model(os.path.join(os.path.join("experiments",
                                                                    experiment_name),
                                            box_preset,
                                            model_arch),
                                            full_path=True)
    
    # Get the predictions from the model path
    y_hats, y_trues, filenames = inference(saved_model_path)

    accs = get_top_k_accuracies(y_hats, y_trues, k_max=top_ks)
    
    return accs, y_hats
    







if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--model_arch", type=str, default="resnet34", help="Model architecture")
    parser.add_argument("--gpu", type=int, default=7, help="GPU to use")
    parser.add_argument("--box_preset", type=str, default="t1-t5", help="Box preset to use")
    parser.add_argument("--experiment_name", type=str, default="base_experiment", help="Name of the experiment")
    args = parser.parse_args()

    saved_model_path = get_the_best_model(os.path.join(os.path.join("experiments",
                                                                    args.experiment_name), 
                                            args.box_preset, 
                                            args.model_arch),
                                            full_path=True)
    
    print(saved_model_path)

    y_hats, y_trues, filenames = inference(saved_model_path)

    print(y_hats.shape, type(y_hats))
    print(y_trues.shape, type(y_trues))
    print(len(filenames), type(filenames))




    accs = get_top_k_accuracies(y_hats, y_trues, k_max=5)

    plt.plot(list(range(1, 6)), accs, marker="*", linewidth=2, markersize=10, label=args.model_arch)
    plt.xlabel("K-th Order")
    plt.ylabel("Accuracy")
    plt.xticks(list(range(1, 6)))
    plt.legend()
    plt.show()
    

    print(accs)












