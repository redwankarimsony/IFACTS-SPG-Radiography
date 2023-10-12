# summarize.py
# Created by sonymd at 3/15/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU



import os
import pandas as pd
import pickle
from accuracy_top_k import select_the_best_model, inference
from configs.config import configClass



def generate_train_summary(results_dir="./temporal_sort", 
                           models = ["resnet34",],
                           box_presets = [0, 1, 2, 3]):
    df = pd.DataFrame()
    df["box_preset"] = box_presets

    def find_best_model(model_dir):
        files = os.listdir(model_dir)
        ckpts = [float("0."+x.split("=")[-1].split(".")[1]) for x in files if x.endswith(".ckpt")]
        return max(ckpts)

    for model_arch in models:
        res = []
        for box in df["box_preset"]:
            model_dir = os.path.join(results_dir, f"box_{box}", model_arch)
            if os.path.exists(model_dir):
                # print(box, model_arch, find_best_model(model_dir))

                res.append(find_best_model(model_dir=model_dir))
            else:
                res.append(0.0)

        df[model_arch] = res


    df.to_csv(os.path.join(results_dir, "top-1-summary.csv"))

    return df


def generate_top_k_acc_summary(results_dir="./saved_iii", 
                           models = ["resnet34",],
                           box_presets = [0, 1, 2, 3],
                           acc_order = 2):
    
    df = pd.DataFrame()
    df["box_preset"] = box_presets
    results = {}
    all_pred_scores= {}
    for model_arch in models:
        for box in box_presets:
            model_dir = os.path.join(results_dir, f"box_{box}", model_arch)
            if os.path.exists(model_dir):
                best_model_path = select_the_best_model(save_dir=results_dir,
                                                        model_arch=model_arch,
                                                        box_preset=box)
                
                cfg = configClass()
                cfg.model_arch = model_arch
                cfg.cuda_devices = [7, ]
                cfg.box_preset = box

                acc_ks, gt, all_scores = inference(cfg, 
                                       best_model_checkpoint=best_model_path,
                                       acc_order=acc_order)
                results[(model_arch, box)] = acc_ks
                all_pred_scores[(model_arch, box)] = all_scores
                if "gt" not in all_pred_scores.keys():
                    print(type(gt))
                    all_pred_scores[("gt", 0)] = gt
            else:
                print(model_arch, box , "is not there yet")
        # df[model_arch] = res

        # df.to_csv(os.path.join(results_dir, f"top-{acc_order}-summary.csv"), index=False)
    with open(os.path.join(results_dir, f"all_acc_upto_rank_10.pkl"), "wb") as fp:
        pickle.dump(results, fp)

    with open(os.path.join(results_dir, f"all_model_scores.pkl"), "wb") as fp:
        pickle.dump(all_pred_scores, fp)

    return df

    
    
    
if __name__ == "__main__":

    results_dir = "temporal_sort"
    models = ["resnet34", "resnet50", "resnet101", "densenet121", "densenet161", "densenet169", "densenet201", 
              "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]

    summary = generate_train_summary(results_dir=results_dir,
                           models=models,
                           box_presets=[0,1,2,3])
    
    print(summary)

    sum_k = generate_top_k_acc_summary(results_dir=results_dir,
                                       models=models,
                                       box_presets=[0,1,2,3],
                                       acc_order=7)


    









        
