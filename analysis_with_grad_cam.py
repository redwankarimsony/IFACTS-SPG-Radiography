#       analysis_with_grad_cam.py
#       Created by Redwan Sony (sonymd) at 10/24/23
#       Mail: sonymd@msu.edu
#       GitHub: www.github.com/redwankarimsony
#       Graduate Researcher, iPRoBe Lab, CSE, MSU


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from train_base import LightningTrainer, get_datasets


def overlay_cam_on_image(img_rgb: np.ndarray,
                         saliency_map: np.ndarray,
                         use_rgb: bool = False,
                         colormap: int = cv2.COLORMAP_JET,
                         image_weight: float = 0.5) -> np.ndarray:
    """
    Overlays the saliency map on the image.
    :param img: The base image in RGB or grayscale.
    :param saliency_map: The saliency map with the same width and height as the image.
    :param use_rgb: If true, converts the saliency map to RGB format.
    :param colormap: The OpenCV colormap to be used to convert the saliency map to RGB format.
    :param image_weight: The weight of the input image.
    :return: The final image with the saliency map overlaid.
    """

    # Apply the colormap to the saliency map
    saliency_map = cv2.applyColorMap(np.uint8(255 * saliency_map), colormap)

    # Normalize the saliency map
    if use_rgb:
        saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2RGB)
    saliency_map = saliency_map.astype(float) / 255

    # Compare the spatial resolution of the heatmap and the input image and resize the heatmap if necessary
    if img_rgb.shape[0:2] != saliency_map.shape[0:2]:
        saliency_map = cv2.resize(saliency_map, img_rgb.shape[0:2], 
                                  interpolation=cv2.INTER_CUBIC)
    
    # Check the image consistency
    if img_rgb.max() > 1.0:
        raise Exception("The input image should np.float32 in the range [0, 1]")
    
    # Check the weight factor
    if not 0 <= image_weight <= 1:
        raise Exception("The weight factor should be between 0 and 1 but\
                         got {}".format(image_weight))
    
    # Overlay the saliency map on the input image
    output = (1-image_weight) * img_rgb + image_weight * saliency_map
    
    # Squeeze the pixel values to be between 0 and 1
    output = output / output.max()
    
    return np.uint8(255 * output)


def denormalize(tensor, mean, std, numpy=True):
    """
    Denormalize a tensor.
    Args:
    - tensor: The normalized tensor.
    - mean: The mean used for normalization.
    - std: The standard deviation used for normalization.
    Returns:
    - The denormalized tensor.
    """
    # Denormalize
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # Convert to channel-last format
    tensor = tensor.permute(1, 2, 0)
    
    # Convert to desired data type
    if numpy:
        return np.uint8(255* tensor.numpy())
    else:
        return tensor.numpy()
    

if __name__ == "__main__":
    # Define the transformations you want to apply to the images
    # best_model_checkpoint = "experiments/base_experiment/t1-t5/resnet34/resnet34_t1-t5_epoch=956-val_acc_epoch=0.7695.ckpt"
    best_model_checkpoint = 'experiments/base_experiment/complete-vertebrae/resnet50/resnet50_complete-vertebrae_epoch=1012-val_acc_epoch=0.5195.ckpt'

    # Load the model
    classifier = LightningTrainer.load_from_checkpoint(best_model_checkpoint)
    classifier.eval()



    # Load the datasets
    ds_train, ds_valid, dl_train, dl_valid = get_datasets(classifier.cfg)

    # Load a single instance from the dataset
    data_idx = 10
    image_torch, ground_truth_class, image_filename = ds_valid[data_idx]
    print(image_torch.shape, ground_truth_class, image_filename)

    # Get The image file path
    orig_img_path = os.path.join(classifier.cfg.dataset_root, 
                            classifier.cfg.box_preset,
                            "valid", 
                            image_filename)


    # Load the image in RGB format
    img = cv2.cvtColor(cv2.imread(orig_img_path), code=cv2.COLOR_BGR2RGB)
    print("Original Input Size:", img.shape)
    print(img.min(), img.max(), type(img))


    img_denorm = denormalize(image_torch, classifier.cfg.stat_mean, classifier.cfg.stat_std)

    print("Denormalized Input Size:", img_denorm.shape)
    print(img_denorm.min(), img_denorm.max(), type(img_denorm))


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)

    plt.imshow(img) 
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img_denorm)
    plt.title("Denormalized Image")
    plt.savefig("denormalized_image.png")


    



# # Resize the image to 512*512
# img_rgb = cv2.resize(img, (512, 512))

# # Convert it into np.float32 between [0,1]
# img_rgb = np.float32(img_rgb) / 255.


# model = classifier.model
# target_layers = [model.layer4[-1]]
# input_tensor = X.unsqueeze(0)
# print(input_tensor.shape)

# # Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# targets = [ClassifierOutputTarget(y)]



# # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# # In this example grayscale_cam has only one image in the batch:
# grayscale_cam = grayscale_cam[0, :]

# print("Saliency Map details Shape:", grayscale_cam.shape, 
#       "Min:", grayscale_cam.min(), 
#       "Max:", grayscale_cam.max())

# visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True, image_weight=0.9)

# cv2.imwrite("grad_cam.png", visualization)

# import matplotlib.pyplot as plt
# plt.imshow(grayscale_cam, cmap='gray')

# plt.savefig("grayscale_cam.png")

# plt.close()
