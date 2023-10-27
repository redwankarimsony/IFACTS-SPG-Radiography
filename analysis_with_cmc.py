#       analysis.py
#       Created by Redwan Sony (sonymd) at 10/24/2023
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import argparse
from inference import get_model_summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from summarize import summarize_single_experiment


# Generate 12 markers
markers = ["*", "o", "v", "s", "p", "P", "h", "H", "D", "d", "X", "x"]

def generate_model_wise_top_k_acc_plot(experiment_name:str, models, model_family, top_ks, display=False, markers = markers):
    """_summary_

    Args:
        experiment_name (_type_): Name of the experiment
        models (_type_): list of model archictectures to be compared
        model_family (_type_): Family of models being compared
        markers (_type_): plot markers
        top_ks (_type_): top-k accuracies to be plotted
        display (bool, optional): If true, it will show the plot at the end of calculation. Defaults to False.
    """
    # Make plot_dir if there is not any
    plot_dir = os.path.join("plots", experiment_name, "cmc_curves")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Get the best saved model from the summary file
    df = pd.read_csv(os.path.join("experiments", experiment_name, "summary", "summary.csv"))
    

    fig = plt.figure(figsize=(10, 6), dpi=200)

    # Get the best model for each model
    for idx, model in enumerate(models):
        df_temp = df[df["model_name"]==model].iloc[0]
        best_model_path = os.path.join("experiments", experiment_name, 
                                    df_temp["box_preset"], 
                                    model, 
                                    df_temp["saved_model"])

        print("Model found at: ", best_model_path) if os.path.exists(best_model_path) else print("Model not found at: ", best_model_path)   

        # Get the model summary
        top_acc, y_preds = get_model_summary(experiment_name=experiment_name,
                                            model_arch=model,
                                            box_preset=df_temp["box_preset"],
                                            top_ks=top_ks,
                                            gpu=2)
        
        # Plot the top-5 accuracies
        plt.plot(list(range(1, top_ks+1)), 
                top_acc, 
                label=f"{model}, {df_temp['box_preset']}", 
                linewidth=2, 
                marker=markers[idx], markersize=10)

    plt.legend()
    plt.xticks(list(range(1, top_ks+1)))
    plt.xlabel("K-th Order")
    plt.ylabel("Accuracy")
    plt.title(f"Top-{top_ks} Accuracies of {model_family} Models")
    plt.savefig(os.path.join(plot_dir, f"{model_family}_top_{top_ks}_accs.png"))
    print("\n\nPlot saved to: ", os.path.join(plot_dir, f"{model_family}_top_{top_ks}_accs.png\n\n"))
    if display:
        plt.show()




def generate_CMC_for_top_models(experiment_name, top_ks=5,  plot_best_models=7, markers=markers):
    """_summary_
    Args:
        experiment_name (_type_): _description_
        top_ks (int, optional): _description_. Defaults to 5.
        plot_best_models (int, optional): _description_. Defaults to 7.
        markers (_type_, optional): _description_. Defaults to markers.
    """

    # Make plot_dir if there is not any
    plot_dir = os.path.join("plots", experiment_name, "cmc_curves")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load the summary file
    df = pd.read_csv(os.path.join("experiments", experiment_name, "summary", "summary.csv"))
    df_top_5 = df.iloc[:plot_best_models]


    fig = plt.figure(figsize=(10, 6), dpi=200)

    # Enumerate each row and get the model summary
    for idx, row in df_top_5.iterrows():
        # Get the model summary
        top_acc, y_preds = get_model_summary(experiment_name=experiment_name,
                                            model_arch=row["model_name"],
                                            box_preset=row["box_preset"],
                                            top_ks=top_ks,
                                            gpu=2)
        
        # Plot the top-5 accuracies
        plt.plot(list(range(1, top_ks+1)), 
                top_acc, 
                label=f"{row['model_name']}, {row['box_preset']}", 
                linewidth=2, 
                marker=markers[idx], markersize=10) 
        
    plt.legend(loc='lower right')
    plt.xticks(list(range(1, top_ks+1)))
    plt.xlabel("K-th Order")
    plt.ylabel("Accuracy")
    plt.title(f"Top-{top_ks} Accuracies of Best Models")
    plt.savefig(os.path.join(plot_dir, f"top_{top_ks}_accs_for_best_{plot_best_models}_models.png"))
    plt.gcf().clear()
    print("\n\nPlot saved to: ", os.path.join(plot_dir, f"top_{top_ks}_accs_for_best_{plot_best_models}_models.png\n\n"))




def generate_CMC_for_top_models_with_box_preset(experiment_name, take_top=3, top_ks=5, markers=markers):
    # Make plot_dir if there is not any
    plot_dir = os.path.join("plots", experiment_name, "cmc_curves")
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.read_csv(os.path.join("experiments", experiment_name, "summary", "summary.csv"))
    df_t1_t5 = df[df["box_preset"]=="t1-t5"].reset_index(drop=True).iloc[:take_top]
    df_clavicle = df[df["box_preset"]=="clavicle-only"].reset_index(drop=True)[:take_top]
    df_complete = df[df["box_preset"]=="complete-vertebrae"].reset_index(drop=True)[:take_top]


    # cobine the dataframes
    df_combined = pd.concat([df_t1_t5, df_clavicle, df_complete], axis=0).reset_index(drop=True)
    



    fig = plt.figure(figsize=(10, 6), dpi=200)
    for idx, row in df_combined.iterrows():
        # Get the model summary
        top_acc, y_preds = get_model_summary(experiment_name="base_experiment",
                                            model_arch=row["model_name"],
                                            box_preset=row["box_preset"],
                                            top_ks=top_ks,
                                            gpu=2)
        
        # Plot the top-5 accuracies
        plt.plot(list(range(1, top_ks+1)), 
                top_acc, 
                label=f"{row['model_name']}, {row['box_preset']}", 
                linewidth=2, 
                marker=markers[idx], markersize=10)
        
    plt.legend(loc='lower right')
    plt.xticks(list(range(1, top_ks+1)))
    plt.xlabel("K-th Order")
    plt.ylabel("Accuracy")
    plt.title(f"Top-5 Accuracies for best 3 models of each box preset")
    plt.savefig(os.path.join(plot_dir, f"top_{top_ks}_accs_for_best_models_with_each_preset.png"), dpi=200)
    plt.gcf().clear()
    print("\n\nPlot saved to: ", os.path.join(plot_dir, f"top_5_accs_for_best_models_with_each_preset.png\n\n"))




def generate_CMC_single_model_multiple_boxes(experiment_name, model_arch, box_presets, top_ks=5):
        # Make plot_dir if there is not any
        plot_dir = os.path.join("plots", experiment_name, "cmc_curves")
        os.makedirs(plot_dir, exist_ok=True)

        fig = plt.figure(figsize=(10, 6), dpi=200)
        for idx, box_preset in enumerate(box_presets):
            # Get the model summary
            top_acc, y_preds = get_model_summary(experiment_name=experiment_name,
                                                model_arch=model_arch,
                                                box_preset=box_preset,
                                                top_ks=top_ks,
                                                gpu=2)
            
            # Plot the top-5 accuracies
            plt.plot(list(range(1, top_ks+1)), 
                    top_acc, 
                    label=f"{model_arch}, {box_preset}", 
                    linewidth=2, 
                    marker=markers[idx], markersize=10)
            
        plt.legend(loc='lower right')
        plt.xticks(list(range(1, top_ks+1)))
        plt.xlabel("K-th Order")
        plt.ylabel("Accuracy")
        plt.title(f"Top-{top_ks} Accuracies for {model_arch} with different box presets")
        plt.savefig(os.path.join(plot_dir, f"top_5_accs_for_{model_arch}_with_different_box_presets.png"), dpi=200)
        plt.gcf().clear()
        print("\n\nPlot saved to: ", os.path.join(plot_dir, f"top_5_accs_for_{model_arch}_with_different_box_presets.png\n\n"))

        

    








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="base_experiment", 
                        choices=["base_experiment",
                                 "augmentation_no_crop"])
    parser.add_argument("--top_ks", type=int, default=5, help="Top-k accuracies to be plotted")
    parser.add_argument("--plot_best", type=int, default=7, help="Number of best models to be plotted")
    parser.add_argument("--take_top", type=int, default=3, 
                        help="Number of best models to be plotted for each box preset")
    parser.add_argument("--box_preset", type=str, default="best", 
                        choices=["t1-t5", "clavicle-only", "complete-vertebrae", "best", "all"], 
                        help="Box preset to be used for generating the CMC curve")
    parser.add_argument("--model_arch", type=str, default="all",
                        choices=["resnet34", "resnet50", "resnet101", "densenet121", "densenet161", "densenet169", "densenet201",
                                 "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
                                 "efficientnet_b5", "efficientnet_b6", "efficientnet_b7", "resnets", "densenets", "efficientnets", "all"],
                                 help="Model architecture to be used for generating the CMC curve")
    args = parser.parse_args()


    # Summarize the results
    summarize_single_experiment(experiment_name=f"{args.experiment_name}", verbose=True)


    resnets = ["resnet34", "resnet50", "resnet101"]
    densenets = ["densenet121", "densenet161", "densenet169", "densenet201"]
    efficentnets = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", 
                    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]

    # Generate the top-5 accuracies of all the resnet models
    if args.box_preset == "best":

        if args.model_arch == "resnets":
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=resnets,
                                            model_family="ResNet",
                                            top_ks=args.top_ks)
            
        elif args.model_arch == "densenets":
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=densenets,
                                            model_family="DenseNet",
                                            top_ks=args.top_ks)
        elif args.model_arch == "efficientnets":
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=efficentnets,
                                            model_family="EfficientNet",
                                            top_ks=args.top_ks)
            
        elif args.model_arch == "all":
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=resnets,
                                            model_family="ResNets",
                                            top_ks=args.top_ks)
            
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=densenets,
                                            model_family="DenseNets",
                                            top_ks=args.top_ks)
            
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=efficentnets,
                                            model_family="EfficientNets",
                                            top_ks=args.top_ks)
            
            
        else:
            generate_model_wise_top_k_acc_plot(experiment_name=args.experiment_name,
                                            models=[args.model_arch,],
                                            model_family=args.model_arch,
                                            markers=["*"],
                                            top_ks=args.top_ks)
        

        
    
    # Generate The CMC curve for the top performing models
    generate_CMC_for_top_models(experiment_name=args.experiment_name,
                                top_ks=args.top_ks,
                                plot_best_models=args.plot_best)


    # Generate the CMC curve for the top N models of each box preset
    generate_CMC_for_top_models_with_box_preset(experiment_name=args.experiment_name,
                                                take_top=3)

    
    # Generate the CMC curve for a single model with different box presets
    generate_CMC_single_model_multiple_boxes(experiment_name=args.experiment_name,
                                            model_arch=args.model_arch,
                                            box_presets=["t1-t5", 
                                                         "clavicle-only", 
                                                         "complete-vertebrae"],
                                            top_ks=args.top_ks)
    



    
    

