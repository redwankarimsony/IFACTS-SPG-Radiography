#       inference.py
#       Created by Redwan Sony (sonymd) at 10/27/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU



import os.path as osp
import pickle
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from inference import inference, get_top_k_accuracies
from itertools import combinations
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def generate_all_logit_predictions(experiment_name="base_experiment"):

    df = pd.read_csv(osp.join("experiments" ,experiment_name, "summary", "summary.csv"))

    # Define a dictionary to store the predictions with keys touple of (model_name, box_preset)
    # Load already saved predictions
    try:
        logit_predictions = load_saved_predictions(experiment_name=experiment_name, verbose=True)
    except:
        logit_predictions = dict()

    # Iterate each rows of the dataframe    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Get the model name
        model_name = row["model_name"]
        # Get the box preset
        box_preset = row["box_preset"]
        # Get the checkpoint file

        # Get the checkpoint directory
        checkpoint_dir = osp.join("experiments",
                                experiment_name,
                                box_preset,
                                model_name,
                                row['saved_model'])
        
        # If the inference is not already done for this model, do it
        try:
            logit_predictions[(model_name, box_preset)]
            print("Inference already done for {}".format((model_name, box_preset)))
        except:
            print("Doing inference for {}".format((model_name, box_preset)))
            # Do the inference
            logit_predictions[(model_name, box_preset)] = inference(saved_model_path=checkpoint_dir, 
                                                                logits=True, map_location="cuda:0")
        
        
        # Save the predictions
        with open(osp.join("experiments", experiment_name, "summary", "logit_predictions.pkl"), "wb") as f:
            pickle.dump(logit_predictions, f)
            print("Saved the predictions to {}".format(osp.join("experiments", experiment_name, "summary", "logit_predictions.pkl")))



def load_saved_predictions(experiment_name="base_experiment", verbose=True):
    with open(osp.join("experiments", experiment_name, "summary", "logit_predictions.pkl"), "rb") as f:
        logit_predictions = pickle.load(f)

    if verbose:
        print("Loaded the predictions from {}".format(osp.join("experiments", experiment_name, "summary", "logit_predictions.pkl")))
        # Print loaded models
        for key in logit_predictions.keys():
            print(key)
    return logit_predictions


def make_fusions(experiment_name='base_experiment', fuse_top=6):
    df = pd.read_csv(osp.join("experiments" ,experiment_name, "summary", "summary.csv"))[:fuse_top]

    # Load the predictions
    logit_predictions = load_saved_predictions(experiment_name=experiment_name, verbose=False)

    # Get the model keys:
    chosen_keys = [(row['model_name'], row['box_preset']) for idx, row in df.iterrows()]
    print(chosen_keys)

    keys_idx = list(range(len(chosen_keys)))
    combinations_idx = list(combinations(keys_idx, 3))
    print(combinations_idx)

    all_fusions = []

    # Fuse the combinations
    for combination_idx in combinations_idx:
        
        
        y_hats, keys = [], []
        for idx in combination_idx:
            keys.append(chosen_keys[idx][0])
            y_hats.append(logit_predictions[chosen_keys[idx]][0])
        
        y_hats_comb = sum(y_hats)/len(y_hats)
        y_trues = logit_predictions[chosen_keys[0]][1]
        all_filenames = logit_predictions[chosen_keys[0]][2]

        accs = get_top_k_accuracies(F.softmax(y_hats_comb, dim=1), y_trues, k_max=5)
        
        # print(tuple(keys), '\t\t', accs[0])
        all_fusions.append((tuple(keys), accs[0]))

    # Sort all fusions based on the keys
    all_fusions = sorted(all_fusions, key=lambda x: x[1], reverse=True)
    
    for comb_models, accuracy in all_fusions:
        print(comb_models, '\t\t', accuracy)



    # plt.bar([comb_models[0] for comb_models, accuracy in all_fusions], 
    #          [accuracy for comb_models, accuracy in all_fusions], marker="*", linewidth=2, markersize=10, label="Fusion")
    labels = []
    for comb_models, accuracy in all_fusions[:5]:
        a = []
        for comb_model in comb_models:
            a.append(comb_model.replace("efficientnet","E").replace("resnet", "R").replace("densenet", "D"))
        labels.append("\n".join(a))
                                                                         
    plt.bar(labels, 
             [accuracy for comb_models, accuracy in all_fusions[:5]])
    plt.xticks()
    plt.ylim([0.8, .9])
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.show()

    

    


if __name__ == "__main__":
    # generate_all_logit_predictions
    # generate_all_logit_predictions()

    # make_fusions
    make_fusions(experiment_name='base_experiment', fuse_top=6)

