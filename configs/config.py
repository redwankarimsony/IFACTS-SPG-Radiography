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
                 finetune_model=False):

        # Basic run parameters
        self.model_arch = model_arch
        self.box_preset = box_preset
        self.cuda_devices = [gpu,]
        self.finetune_model = finetune_model

        # Experiment paths
        self.prefix = "experiments"
        self.experiment_name = "base_experiment"
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
        self.num_workers = min(self.batch_size, int(os.cpu_count() * 0.8))
        self.pin_memory = True
        self.use_cache = True
        self.limit_train_batches = 250
        self.max_epoch = 1200
        self.learning_rate = 1e-4

    def _setup_data_transformations(self):
        # ImageNet statistics as default
        self.stat_mean = [0.485, 0.456, 0.406]
        self.stat_std = [0.229, 0.224, 0.225]

        self.tfms_train = tf.Compose([
            tf.ToTensor(),
            tf.Resize(self.width),
            tf.CenterCrop(self.width),
            tf.RandomRotation(degrees=8),
            tf.RandomPerspective(distortion_scale=0.2, p=0.3),
            tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.3),
            tf.Normalize(mean=self.stat_mean, std=self.stat_std)
        ])

        self.tfms_valid = tf.Compose([
            tf.ToTensor(),
            tf.Resize(self.width),
            tf.CenterCrop(self.width),
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
    
         
        
