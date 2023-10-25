#       analysis_failures.py
#       Created by Redwan Sony (sonymd) at 10/24/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import torch
import pandas as pd
from train_base import LightningTrainer, get_datasets
from inference import inference


def get_failure_cases(best_model_path, save=False):
    # Get Model location:
    
    y_hats, y_trues, filenames = inference(saved_model_path=best_model_path, map_location="cuda:1")
    y_preds = y_hats.argmax(dim=1)
        

    # Get the dataset and the label2id mappings
    cfg = LightningTrainer.load_from_checkpoint(best_model_path).cfg
    ds_train, ds_valid, dl_train, dl_valid = get_datasets(cfg)
    label2id = ds_valid.label2id


    # Make a dataframe of the failure cases
    df = pd.DataFrame({"Filenames": filenames, "TrueID": y_trues, "PredictedID": y_preds, })
    df["TrueID"] = df["TrueID"].apply(lambda x: label2id[x])
    df["PredictedID"] = df["PredictedID"].apply(lambda x: label2id[x])

    # Drop all the rows which has y_true == y_pred
    df = df[df["TrueID"] != df["PredictedID"]]
    df.reset_index(inplace=True, drop=True)

    if save:
        try:
            model_location, filename = os.path.split(best_model_path)
            new_filename = filename.replace(".ckpt", ".xlsx")
            df.to_excel(os.path.join(model_location, new_filename), index=False)
            print("Failure cases saved to ", os.path.join(model_location, new_filename))
        except Exception as e:
            print(e)
            print("Failure cases could not be saved.")

    return df



if __name__ == "__main__":
    best_model_path = "experiments/base_experiment/t1-t5/densenet161/densenet161_t1-t5_epoch=541-val_acc_epoch=0.8122.ckpt"

    df = get_failure_cases(best_model_path, save=True)

    
    # for y_pred, y_true, filename  in zip(y_preds, y_trues, filenames):
    #     if y_true != y_pred:
    #         print(filename, "Original: ", label2id[y_true.item()], "Predicted: ", label2id[y_pred.item()])


    print(df.head())





