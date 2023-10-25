# <center>IFACTS-SPG-RADIOGRAPHY</center>

**How to Run Experiments**

1. Make a new config file in the `configs` directory appropriately
2. Update the configuration import in the `train_base.py` file to make things smoother.
3. Run your bash script witht the appropriate commands. Care should be taken to select the gpus concurrent processes delay time etc in the bash script. 
4. After all the experiment have finished running, run the `summarize.py` file with the appropriate experiment name to summarize that experiment only in the `experiments` directory. 
5. Use the script `analysis_with_cmc.py` to generate the cumulative matching curves for the models and box presets. 
6. Use the script `analysis_with_grad_cam.py` to generate the gradCAM analysis of the classification systems. 

