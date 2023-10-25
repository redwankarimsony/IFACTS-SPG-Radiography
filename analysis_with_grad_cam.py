#       accuracy_top_k.py
#       Created by Redwan Sony (sonymd) at 10/24/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from train_base import LightningTrainer, get_datasets



best_model_checkpoint = "experiments/base_experiment/t1-t5/resnet34/resnet34_t1-t5_epoch=956-val_acc_epoch=0.7695.ckpt"

# Load the model
classifier = LightningTrainer.load_from_checkpoint(best_model_checkpoint)
classifier.eval()


# Load the datasets
ds_train, ds_valid, dl_train, dl_valid = get_datasets(classifier.cfg)

data_idx = 10
X, y, code = ds_valid[data_idx]

print(f"TargetClass: {code} ({y}) {classifier.cfg.box_preset}")
img_path = os.path.join(classifier.cfg.dataset_root, 
                        classifier.cfg.box_preset,
                        "valid", 
                        code)   

print(img_path, os.path.exists(img_path))

img = cv2.imread(img_path)
print(img.shape)

# Resize the image to 512*512
img_rgb = cv2.resize(img, (512, 512))

# Convert it into np.float32 between [0,1]
img_rgb = np.float32(img_rgb) / 255.


model = classifier.model
target_layers = [model.layer4[-1]]
input_tensor = X.unsqueeze(0)
print(input_tensor.shape)

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

targets = [ClassifierOutputTarget(y)]



# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]

print("Saliency Map details Shape:", grayscale_cam.shape, 
      "Min:", grayscale_cam.min(), 
      "Max:", grayscale_cam.max())

visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True, image_weight=0.9)

cv2.imwrite("grad_cam.png", visualization)

import matplotlib.pyplot as plt
plt.imshow(grayscale_cam, cmap='gray')

plt.savefig("grayscale_cam.png")

plt.close()
