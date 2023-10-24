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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ResNet <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Generate the top-5 accuracies of all the resnet models


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




if __name__ == "__main__":
    # Generate the top-5 accuracies of all the resnet models
    generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
                                        models=["resnet34", "resnet50", "resnet101"],
                                        model_family="ResNet",
                                        markers=["*", "o", "v"],
                                        top_ks=5)
      


    # Generate the top-5 accuracies of all the densenet models
    generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
                                        models=["densenet121", "densenet161", "densenet169", "densenet201"],
                                        model_family="DenseNet",
                                        markers=["*", "o", "v", "s"],
                                        top_ks=5)
    


    # Generate the top-5 accuracies of all the efficientnet models

    generate_model_wise_top_k_acc_plot(experiment_name="base_experiment",
                                        models=["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"],
                                        model_family="EfficientNet",
                                        markers=["*", "o", "v", "s", "p", "P", "h", "H"],
                                        top_ks=5)
    


    
    

