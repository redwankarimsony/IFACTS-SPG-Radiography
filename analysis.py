#       analysis.py
#       Created by Redwan Sony (sonymd) at 10/24/2023
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU



import os
from inference import get_model_summary
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def generate_model_wise_top_k_acc_plot(experiment_name, models, model_family, markers, top_ks, display=False ):
    """_summary_

    Args:
        experiment_name (_type_): Name of the experiment
        models (_type_): list of model archictectures to be compared
        model_family (_type_): Family of models being compared
        markers (_type_): plot markers
        top_ks (_type_): top-k accuracies to be plotted
        display (bool, optional): If true, it will show the plot at the end of calculation. Defaults to False.
    """
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
    plt.title(f"Top-5 Accuracies of {model_family} Models")
    plt.savefig(os.path.join("plots", experiment_name, f"{model_family}_top_{top_ks}_accs.png"))
    if display:
        plt.show()



def generate_CMC_for_top_models(top_ks=5, plot_best = 7):
    # Load the summary file
    
    experiment_name = "base_experiment"
    df = pd.read_csv(os.path.join("experiments", experiment_name, "summary", "summary.csv"))
    df_top_5 = df.iloc[:plot_best]
    markers = ["*", "o", "v", "s", "p", "P", "h", "H"]


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
    plt.title(f"Top-5 Accuracies of Best Models")

    plt.savefig(os.path.join("plots", experiment_name, f"top_{top_ks}_accs_for_best_models.png"))
    plt.show()



def generate_CMC_for_top_models_with_box_preset(take_top=3):
    df = pd.read_csv(os.path.join("experiments", "base_experiment", "summary", "summary.csv"))
    df_t1_t5 = df[df["box_preset"]=="t1-t5"].reset_index(drop=True).iloc[:take_top]
    df_clavicle = df[df["box_preset"]=="clavicle-only"].reset_index(drop=True)[:take_top]
    df_complete = df[df["box_preset"]=="complete-vertebrae"].reset_index(drop=True)[:take_top]


    # cobine the dataframes
    df_combined = pd.concat([df_t1_t5, df_clavicle, df_complete], axis=0).reset_index(drop=True)
    

    # Generate 12 markers
    markers = ["*", "o", "v", "s", "p", "P", "h", "H", "D", "d", "X", "x"]

    fig = plt.figure(figsize=(10, 6), dpi=200)
    for idx, row in df_combined.iterrows():
        # Get the model summary
        top_acc, y_preds = get_model_summary(experiment_name="base_experiment",
                                            model_arch=row["model_name"],
                                            box_preset=row["box_preset"],
                                            top_ks=5,
                                            gpu=2)
        
        # Plot the top-5 accuracies
        plt.plot(list(range(1, 6)), 
                top_acc, 
                label=f"{row['model_name']}, {row['box_preset']}", 
                linewidth=2, 
                marker=markers[idx], markersize=10)
        
    plt.legend(loc='lower right')
    plt.xticks(list(range(1, 6)))
    plt.xlabel("K-th Order")
    plt.ylabel("Accuracy")
    plt.title(f"Top-5 Accuracies for best 3 models of each box preset")
    plt.savefig(os.path.join("plots", "base_experiment", f"top_5_accs_for_best_models_with_each_preset.png"), dpi=200)
    plt.show()






    








if __name__ == "__main__":
    resnets = ["resnet34", "resnet50", "resnet101"]
    densenets = ["densenet121", "densenet161", "densenet169", "densenet201"]
    efficentnets = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", 
                    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]

#     # Generate the top-5 accuracies of all the resnet models
#     generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
#                                         models=resnets,
#                                         model_family="ResNet",
#                                         markers=["*", "o", "v"],
#                                         top_ks=5)
      


#     # Generate the top-5 accuracies of all the densenet models
#     generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
#                                         models=densenets,
#                                         model_family="DenseNet",
#                                         markers=["*", "o", "v", "s"],
#                                         top_ks=5)
    


#     # Generate the top-5 accuracies of all the efficientnet models

#     generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
#                                         models=efficentnets,
#                                         model_family="EfficientNet",
#                                         markers=["*", "o", "v", "s", "p", "P", "h", "H"],
#                                         top_ks=5)
    
    # Generate The CMC curve for the top performing models
    generate_CMC_for_top_models(top_ks=20, plot_best=7)


    # Generate the CMC curve for the top N models of each box preset
    # generate_CMC_for_top_models_with_box_preset(take_top=3)



    
    

