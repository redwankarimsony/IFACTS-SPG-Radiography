# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

import os
import torchvision.transforms as tf

class ConfigClass:
    def __init__(self, 
                 model_arch="resnet34", 
                 box_preset="t1-t5", 
                 gpu=7, 
                 experiment_name ="base_experiment",
                 finetune_model=False):
        self.model_names = [
            "resnet34", "resnet50", "resnet101",
            "densenet121", "densenet161", "densenet169",
            "densenet201", "efficientnet_b0", "efficientnet_b1",
            "efficientnet_b2", "efficientnet_b3", "efficientnet_b4",
            "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
            "vit_large_r50_s32_384", "vit_large_patch32_384", "swin_t",
            "swin_s", "swin_b" ]
        
        self.box_presets = ["t1-t5", "clavicle-only", "complete-vertebrae", "whole"]
                

        # Basic run parameters
        if model_arch in self.model_names:
            
            self.model_arch = model_arch
        else:
            raise Exception(f"Model {model_arch} not found. Please choose from {self.model_names}")
        

        if box_preset in self.box_presets:
            self.box_preset = box_preset
        else:
            raise Exception(f"Box preset {self.box_preset} not found. Please choose from {self.box_presets}")
        
        self.cuda_devices = [gpu,]
        self.finetune_model = finetune_model

        # Experiment paths
        self.prefix = "experiments"
        self.experiment_name = experiment_name
        self.results_dir = os.path.join(self.prefix, self.experiment_name)
        self.checkpoint_dir = os.path.join(self.results_dir, self.box_preset, self.model_arch)




        # Dataset parameters
        self._setup_dataset_parameters()

        # Training parameters
        self._setup_training_parameters()

        # Data transformation
        self._setup_data_transformations()

        # Make the results directory
        self._make_results_dir()

        # Save the config
        self._save_config()




    def _setup_dataset_parameters(self):
        self.dataset_root = "/research/iprobe-sonymd/MSU-SPG-Radiography-Dataset/cropped_images"
        self.dataset_dir = os.path.join(self.dataset_root, self.box_preset)
        self.use_mxrecord = True
        self.width = 512
        self.height = 512
        self.num_classes = 760
        self.cuda_precision = "high"  # Options: ["medium", "high", "highest"]

    def _setup_training_parameters(self):
        self.batch_size = 16
        self.num_workers = min(self.batch_size//2, int(os.cpu_count() * 0.8))
        self.pin_memory = True
        self.use_cache = True
        self.limit_train_batches = 250
        self.max_epoch = 1200
        self.learning_rate = 1e-4

    def _setup_data_transformations(self):
        # ImageNet statistics as default
        self.stat_mean = [0.485, 0.456, 0.406]
        self.stat_std = [0.229, 0.224, 0.225]

        if self.experiment_name == "base_experiment":

            self.tfms_train = tf.Compose([
                tf.ToTensor(),
                tf.Resize(self.width),
                tf.CenterCrop(self.width),
                tf.RandomRotation(degrees=8),
                tf.RandomPerspective(distortion_scale=0.2, p=0.3),
                tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
                tf.Normalize(mean=self.stat_mean, std=self.stat_std)
            ])
        elif self.experiment_name == "augmentation_check":
            self.tfms_train = tf.Compose([
                # Random resizing and cropping
                tf.RandomResizedCrop(size=self.width, scale=(0.8, 1.0)),  # Assuming you want 224x224 images, adjust if needed

                # Small rotations
                tf.RandomRotation(degrees=10),  # Rotate +/- 10 degrees

                # Horizontal flipping
                tf.RandomHorizontalFlip(p=0.5),  # 50% probability of applying the flip

                # Adjust brightness, contrast, and saturation
                tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

                # Random affine transformations
                tf.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),

                # Convert to tensor for model input
                tf.ToTensor(),

                # Normalize (Assuming normalization values. Change this if you have specific mean and std for your dataset)
                tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.experiment_name == "augmentation_no_crop":
            self.tfms_train = tf.Compose([
                tf.ToTensor(),
                tf.Resize((self.height, self.width)),
                tf.RandomRotation(degrees=8),
                tf.RandomPerspective(distortion_scale=0.2, p=0.3),
                tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
                tf.Normalize(mean=self.stat_mean, std=self.stat_std)
            ])

        
        if self.experiment_name in ["base_experiment", "augmentation_check"]:
            self.tfms_valid = tf.Compose([
                tf.ToTensor(),
                tf.Resize(self.width),
                tf.CenterCrop(self.width),
                tf.Normalize(mean=self.stat_mean, std=self.stat_std)
            ])

        elif self.experiment_name == "augmentation_no_crop":
            self.tfms_valid = tf.Compose([
                tf.ToTensor(),
                tf.Resize((self.height, self.width)),d
                tf.Normalize(mean=self.stat_mean, std=self.stat_std)
            ])


    def _make_results_dir(self):
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except Exception as e:
            print(e)

    def _save_config(self):
        with open(os.path.join(self.checkpoint_dir, "config.txt"), "w") as file:
            for attr, val in self.__dict__.items():
                file.write(f"{attr} = {val}\n")

        print(f"Config saved to {self.checkpoint_dir}")



    def __str__(self) -> str:
        configs = ""
        for attr, val in self.__dict__.items():
            configs += f"{attr} = {val}\n"
        return configs
    
         
        
