# summarize.py
# Created by sonymd at 3/15/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU



import os
import pandas as pd
import argparse




def get_the_best_model(checkpoint_dir, ext=".ckpt"):
    """
    Returns the file path of the best model in the given checkpoint directory

    Args:
        checkpoint_dir (_type_): Location of the checkpoint directory whree the models are saved
        ext (str, optional): Extension of the checkpoints file. Defaults to ".ckpt".

    Returns:
        _type_: Returns the checkpoint file with the best file accuracy
    """

    # List all the files in the checkpoint directory with given extension
    files = os.listdir(checkpoint_dir)
    # Get the checkpoint files
    ckpts = [x for x in files if x.endswith(ext)]
    # Return the best model
    return  max(ckpts, key= lambda x: float(x.split("=")[-1].split(".")[0]))




def generate_train_summary(experiment_name:str, box_presets:list, model_names:list):
    """
    Generates a summary of the training results for the given experiment name, box presets and model names
    """

    all_models = []

    for box_preset in box_presets:
        for model_name in model_names:
            checkpoint_dir = os.path.join(experiment_name, box_preset, model_name)
            if os.path.exists(checkpoint_dir):
                best_model = get_the_best_model(checkpoint_dir=checkpoint_dir)
                all_models.append(best_model)
    return all_models


    

def summarize_single_experiment(experiment_name:str, verbose=True):
    """Given the location of the experiment directory, this function summarizes the results of the experiment

    Args:
        experiment_name (str): Location of the experimenets

    Returns:
        _type_: the summary of the best model for each box_preset and model_name
    """


    all_models = []

    # List all the box presets
    box_presets = [x for x in os.listdir(experiment_name) if os.path.isdir(os.path.join(experiment_name, x))]   

    for box_preset in box_presets:
        # List all the models for the given box preset
        model_names = os.listdir(os.path.join(experiment_name, box_preset))

        for model_name in model_names:
            checkpoint_dir = os.path.join(experiment_name, box_preset, model_name)
            try:
                best_model = get_the_best_model(checkpoint_dir=checkpoint_dir)
                all_models.append(best_model)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Could not find the best model for {checkpoint_dir}")

    data = []

    for idx, filepath in enumerate(all_models):
        parts = filepath.split("_")

        # If the model name starts with efficientnet, then the model name is
        if parts[0].startswith("efficientnet"):
            model_name = parts[0] + "_" + parts[1]
            box_preset = parts[2]
        
        # Otherwise, the model name is the first part
        else:
            model_name = parts[0]
            box_preset = parts[1]

        # Find the validation accuracy:
        val_acc  = float(parts[-1].split("=")[-1].replace(".ckpt", ""))
        
        # Print the results
        if verbose:
            if idx == 0:
                    print(f"{'MODEL':<20} {'CATEGORY':<25} {'SCORE'}")
                    print(f"{'-----':<20} {'--------':<25} {'-----'}")
        
            print(f"{model_name:<20} {box_preset:<25} {val_acc:.4f}")

        # Append the results to the data list
        data.append({"model_name": model_name, "box_preset": box_preset, "val_acc": val_acc, "saved_model": filepath})

    # Create a dataframe from the results
    df = pd.DataFrame(data)

    # Sort the dataframe by validation accuracy
    df = df.sort_values(by="val_acc", ascending=False)

    # Save the dataframe to the experiment directory
    df.to_csv(os.path.join(experiment_name, "summary.csv"), index=False, sep=",")

    # Print the dataframe path
    print(f"\n\nSummary saved to {os.path.join(experiment_name, 'summary.csv')}")
    
    return df
    




if __name__ == "__main__":

    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="experiments/base_experiment")
    args = parser.parse_args()

    # models = ["resnet34", "resnet50", "resnet101", "densenet121", "densenet161", "densenet169", "densenet201", 
    #           "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]
    # box_presets = ["t1-t5", "clavicle-only", "complete-vertebrae"]

    
    # Summarize the results
    summarize_single_experiment(experiment_name=f"{args.experiment_name}", verbose=True)
    