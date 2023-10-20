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



from train_base import LightningTrainer, get_datasets
from configs.config import ConfigClass
from dataset_mx import MXNetRecDataset
from utils import get_gpu_with_least_memory_over_period
from summarize import get_the_best_model


torch.set_float32_matmul_precision("high")


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





    #         top_k_labels, top_k_probs = get_predictions(y_hat, k=10)

    #         for label, top_k_label in zip(labels, top_k_labels):
    #             if label.item() in top_k_label.tolist():
    #                 top_k_predictions.append(1.)
    #             else:
    #                 top_k_predictions.append(0.)
                    
    #         all_predictions.extend(top_k_predictions)

    # print(f"Accuracy: {np.mean(all_predictions)}")







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



    for k in range(1, 11):
        # Get top-k predictions
        top_k_labels, top_k_probs = get_top_k_predictions(y_hats, k=k)
    
        top_k_predictions = []
        for y_true, top_k_label in zip(y_trues, top_k_labels):
            if y_true.item() in top_k_label.tolist():
                top_k_predictions.append(1.)
            else:
                top_k_predictions.append(0.)

        acc = torch.mean(torch.tensor(top_k_predictions)).item()

        print(f"Top-{k} Accuracy: {acc}")


    



    
    
    
    
#     
#     print(args)

#     cfg = ConfigClass(**vars(args))
#     print(cfg)


#     # inference(cfg)











