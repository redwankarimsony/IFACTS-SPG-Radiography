#       analysis_failures.py
#       Created by Redwan Sony (sonymd) at 11/13/2023
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU



import os
import seaborn as sns
import torch
import argparse
import pandas as pd
from train_base import LightningTrainer, get_datasets
from inference import inference
from analysis_f1_scores import *
from datasets import ChextXrayVerificationDataset
import torchvision.transforms as tf
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def getTMRatFMR(fpr, tpr, thresholds, target_fmr):
    # Calculate the FPR, TPR, and thresholds
    # fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find the threshold closest to your target FMR (1% in this case)
    target_fmr = 0.01
    closest_index = np.argmin(np.abs(fpr - target_fmr))
    closest_fmr = fpr[closest_index]
    closest_tmr = tpr[closest_index]
    closest_threshold = thresholds[closest_index]

    # print("Closest FMR:", closest_fmr)
    # print("Corresponding TMR:", closest_tmr)
    # print("Threshold at this point:", closest_threshold)

    print(f"TMR: {closest_tmr:0.4f} at FMR: {closest_fmr:0.4f} with threshold: {closest_threshold}")

    return closest_tmr, closest_threshold


def failure_cases_analysis(prediction_file="verification-results/results-verification-densenet161-t1-t5.csv", threshold=0.0165):
    df = pd.read_csv(prediction_file, delimiter=",")


    # Get the false positives
    false_positives = df[(df["y_true"] == 0) & (df["y_pred"] > threshold)]
    false_negatives = df[(df["y_true"] == 1) & (df["y_pred"] < threshold)] 

    # Plot the false positives
    for idx, false_positive in false_positives.iterrows():
        print(false_positive["img1_path_abs"], false_positive["img2_path_abs"], false_positive["y_pred"])

        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.imshow(plt.imread(false_positive["img1_path_abs"].strip()))
        plt.title(false_positive["img1_path_abs"].split("/")[-1])
        plt.subplot(1,2,2)
        plt.imshow(plt.imread(false_positive["img2_path_abs"].strip()))
        plt.title(false_positive["img2_path_abs"].split("/")[-1])

        plt.suptitle(f"Ground Truth: {false_positive['y_true']} Similarity Score: {false_positive['y_pred']:0.4f}")

        plt.savefig(f"verification-results/failure_cases/false_positive_{idx}.png")
        plt.close()


    # Plot the false positives

    for idx, false_negative in false_negatives.iterrows():
        print(false_negative["img1_path_abs"], false_negative["img2_path_abs"], false_negative["y_pred"])

        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        plt.imshow(plt.imread(false_negative["img1_path_abs"].strip()))
        plt.title(false_negative["img1_path_abs"].split("/")[-1])
        plt.subplot(1,2,2)
        plt.imshow(plt.imread(false_negative["img2_path_abs"].strip()))
        plt.title(false_negative["img2_path_abs"].split("/")[-1])

        plt.suptitle(f"Ground Truth: {false_negative['y_true']} Similarity Score: {false_negative['y_pred']:0.4f}")

        plt.savefig(f"verification-results/failure_cases/false_negative_{idx}.png")
        plt.close()



    





def verification_results(predictions_file="verification_results.csv", display_plots=True):

    df = pd.read_csv(predictions_file, delimiter=",")

    # Assuming you have `scores` as your model's output probabilities and `labels` as the true binary labels
    y_preds = df["y_pred"].tolist()
    y_trues = df["y_true"].tolist()

    # Calculate False positive rate (fpr) and True positive rate (tpr)
    fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds)
    auc = metrics.auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xscale('log')  # set x-axis to logarithmic scale
    plt.xlim([max(min(fpr), 1e-3), 1.0])  # set limits for x-axis, avoiding log(0)
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Log Scale)')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("verification-results/roc_curve.png")
    if display_plots:
        plt.show()
    else:
        plt.close()



    # Show the genuine and imposter scores distribution
    genuine_scores = df[df['y_true'] == 1]['y_pred'].tolist()
    imposter_scores = df[df['y_true'] == 0]['y_pred'].tolist()

    plt.figure()
    plt.hist(genuine_scores, bins=50, density=True, alpha=0.7,  label='Genuine')
    plt.hist(imposter_scores, bins=50, density=True, alpha=0.7, label='Imposter')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.savefig("verification-results/score_distribution.png")
    if display_plots:
        plt.show()
    else:
        plt.close()


    # Calculate the FPR, TPR, and thresholds
    TMR, target_threshold = getTMRatFMR(fpr, tpr, thresholds, target_fmr=0.01)

    # Calculate the EER
    eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    with open("verification-results/results-summary.txt", "w") as f:
        print(f"AUC: {auc:0.4f}", file=f)
        print(f"EER: {eer:0.4f}", eer, file=f)
        print(f"EER Threshold: {eer_threshold:0.4f}", file=f)
        print(f"TMR: {TMR:0.4f} at 1% FMR", file=f)
        print(f"Target Threshold: {target_threshold:0.4f}", file=f)

    # plot the confusion matrix
    y_preds = [1 if x > target_threshold else 0 for x in y_preds]
    cm = metrics.confusion_matrix(y_trues, y_preds)
    print(cm)

    # Plot the confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig("verification-results/confusion_matrix.png")
    if display_plots:
        plt.show()
    else:
        plt.close()




@torch.no_grad()
def run_verification_setup(model_checkpoint, device="cuda:0"):
    
    # Load the model
    model = LightningTrainer.load_from_checkpoint(model_checkpoint).to(device)
    model.eval()

    cfg = model.cfg

    # Get the dataset
    dir_pebble7 = "/localscratch2/sonymd/MSU-SPG-Radiography-Dataset/cropped_images"
    dir_pebble6 = "/scratch1/sonymd/MSU-SPG-Radiography-Dataset/cropped_images"

    if os.path.exists(dir_pebble7):
        dataset_root = dir_pebble7
    else:
        dataset_root = dir_pebble6


    tfms_valid= tf.Compose([
            tf.ToTensor(),
            tf.Resize((512, 512)),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    ds = ChextXrayVerificationDataset(data_dir=os.path.join(dataset_root, cfg.box_preset),
                                      split="valid",
                                      transform=tfms_valid)
    
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)


    # Get the cosine similarity
    cosine_similarity_function = torch.nn.CosineSimilarity(dim=1, eps=1e-6) 

    # Generate Predictions
    all_img1_paths, all_img2_paths, all_verification_labels, all_verification_predictions = [], [], [], []

    for img1s, img2s, verification_labels, img_path1s, img_path2s in tqdm(dl, total=len(dl)):
        img1_features = model.model(img1s.to(device))
        img2_features = model.model(img2s.to(device))

        # take the softmax  
        img1_features = torch.nn.functional.softmax(img1_features, dim=1)
        img2_features = torch.nn.functional.softmax(img2_features, dim=1)



        # img1_features = img1_features / img1_features.norm(dim=0)
        # img2_features = img2_features / img2_features.norm(dim=0)

        # Calculate the cosine similarity
        cosine_similarity = cosine_similarity_function(img1_features, img2_features)

        # Save the predictions
        all_img1_paths.extend(img_path1s)
        all_img2_paths.extend(img_path2s)
        all_verification_labels.extend(verification_labels.tolist())
        all_verification_predictions.extend(cosine_similarity.tolist())



    # Normalize the predictions between 0 and 1
    # all_verification_predictions = [(x+1)/2 for x in all_verification_predictions]

    y_trues = all_verification_labels
    y_preds = all_verification_predictions

    with open(f"results-verification-{cfg.model_arch}-{cfg.box_preset}.csv", "w") as f:
        print("img1_path,img2_path,y_true,y_pred, img1_path_abs, img2_path_abs", file=f)
              
        for y_true, y_pred, img1_path, img2_path in zip(y_trues, y_preds, all_img1_paths, all_img2_paths):
            print(img1_path.split("/")[-1] + ',' + img2_path.split('/')[-1] + ',' + str(y_true) + ',' + f"{y_pred:0.4f}" + ',', img1_path + ','+img2_path, file=f)




    
    





if __name__ == "__main__":
    # run_verification_setup(model_checkpoint="/research/iprobe-sonymd/IFACTS-SPG-Radiography/experiments/base_experiment/t1-t5/densenet161/densenet161_t1-t5_epoch=458-val_acc_epoch=0.8475.ckpt") 

    verification_results(predictions_file="verification-results/results-verification-densenet161-t1-t5.csv", display_plots=False)


    failure_cases_analysis(prediction_file="verification-results/results-verification-densenet161-t1-t5.csv")
