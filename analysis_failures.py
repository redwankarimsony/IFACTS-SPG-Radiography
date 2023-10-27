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
from analysis_f1_scores import *


def get_failure_cases(best_model_path, save=False, device="cuda:1"):
    # Get Model location:
    
    y_hats, y_trues, filenames = inference(saved_model_path=best_model_path, map_location=device)
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



def get_detailed_report(y_preds, y_trues):
    report = {}
    report['f1_macro'] = get_macro_f1_score(y_preds=y_preds,
                                           y_trues=y_trues)
    
    report["f1_micro"] = get_micro_f1_score(y_preds=y_preds,
                                           y_trues=y_trues)
    
    report["f1_weighted"] = get_weighted_f1_score(y_preds=y_preds,
                                                    y_trues=y_trues)
    
    report["classification_report"] = get_classification_report(y_preds=y_preds,
                                                                y_trues=y_trues)
    
    return report


if __name__ == "__main__":
    best_model_path = "experiments/augmentation_no_crop/clavicle-only/densenet121/densenet121_clavicle-only_epoch=351-val_acc_epoch=0.8186.ckpt"

    # # Get the failure analysis
    # df = get_failure_cases(best_model_path, save=True)


    # Get the predictions
    y_hats, y_trues, filenames = inference(saved_model_path=best_model_path, map_location="cuda:1")

    # Get the report
    report = get_detailed_report(y_preds=y_hats.argmax(dim=1).numpy(), y_trues=y_trues)
    with open("report.txt", "w") as f:
        f.write(str(report["classification_report"]))
    







