# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU
import os
import torchvision.transforms as tf
from torch.nn import Sequential


class configClass:
    def __init__(self):
        self.experiment_name = 'base_experiment'
        self.results_dir = f"results/{self.experiment_name}"
        self.img_dir = "../Annotation/images_annotation_2"
        self.annot_dir = "../Annotation/Three_ROI_280_identities"
<<<<<<< Updated upstream
        self.width = 384
        self.height = 384
        assert self.width == self.height, "The height and width of the input image should be the same"
        self.limit_train_batches = 28
        self.max_epoch = 500
=======
        self.width = 512
        self.height = 512
        assert self.width == self.height, "Height and width sould always be the same"
        self.limit_train_batches = 64
        self.max_epoch = 1000
>>>>>>> Stashed changes

        self.valid_ratio = 0.5

        # [0=(T1-T5 segment only),
        # 1= (Whole Vertebral Column Only)
        # 2= (T1-T5) with Clavicle
        # 3 =(Consider Whole Xray)]
        self.box_preset = 0

<<<<<<< Updated upstream
        self.model_arch = "vit_large_patch32_384"
=======
        self.model_arch = "efficientnet_b5"
>>>>>>> Stashed changes
        self.num_classes = 281
        self.pin_memory = True
        self.cuda_precision = "highest"  # ["medium", "high", "highest"]
        self.cuda_devices =[4]

        self.stat_mean = [0.485, 0.456, 0.406]
        self.stat_std = [0.229, 0.224, 0.225]

        self.batch_size = 8
        self.num_workers = min(self.batch_size, int(os.cpu_count() * 0.8))
        self.learning_rate = (1e-4)/2

        if self.model_arch.startswith("vit_"):
            self.width = 224
            self.height = 224

        self.tfms_train = Sequential(tf.Resize(self.width),
                                     tf.CenterCrop(self.width),
                                     tf.RandomRotation(degrees=8),
                                     tf.RandomPerspective(distortion_scale=0.2, p=0.3),
                                     tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.6),
                                     tf.Normalize(mean=self.stat_mean, std=self.stat_std))

        self.tfms_valid = Sequential(tf.Resize(self.width),
                                     tf.CenterCrop(self.width),
                                     # tf.RandomRotation(degrees=5),
                                     # tf.RandomAdjustSharpness(sharpness_factor=1.3, p=0.6),
                                     tf.Normalize(mean=self.stat_mean, std=self.stat_std))

        self.make_results_dir()

    def make_results_dir(self):
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except Exception as e:
            print(e)


cfg = configClass()
