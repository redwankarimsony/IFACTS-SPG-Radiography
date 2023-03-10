# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU


import timm
import torch.nn as nn
from torchvision.models import DenseNet121_Weights
from torchvision.models import DenseNet161_Weights
from torchvision.models import DenseNet169_Weights
from torchvision.models import DenseNet201_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import ResNet34_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import EfficientNet_B1_Weights
from torchvision.models import EfficientNet_B2_Weights
from torchvision.models import EfficientNet_B3_Weights
from torchvision.models import EfficientNet_B4_Weights
from torchvision.models import EfficientNet_B5_Weights
from torchvision.models import EfficientNet_B6_Weights
from torchvision.models import EfficientNet_B7_Weights
from torchvision.models import Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
from torchvision.models import densenet
from torchvision.models import efficientnet
from torchvision.models import resnet101
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import swin_t, swin_s, swin_b


def get_model(cfg):
    if cfg.model_arch.startswith("resnet"):
        if cfg.model_arch == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "resnet101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT, progress=True)
        else:
            model = None
            print(f"No ResNet model found with key {cfg.model_arch}")
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch.startswith("densenet"):
        if cfg.model_arch == "densenet121":
            model = densenet.densenet121(DenseNet121_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet161":
            model = densenet.densenet161(DenseNet161_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet169":
            model = densenet.densenet169(DenseNet169_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet201":
            model = densenet.densenet201(DenseNet201_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = None
            print(f"No DenseNet model found with key {cfg.model_arch}")
        model.classifier = nn.Linear(model.classifier.in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch.startswith("efficientnet"):
        if cfg.model_arch == "efficientnet_b0":
            model = efficientnet.efficientnet_b0(EfficientNet_B0_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b1":
            model = efficientnet.efficientnet_b1(EfficientNet_B1_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b2":
            model = efficientnet.efficientnet_b2(EfficientNet_B2_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b3":
            model = efficientnet.efficientnet_b3(EfficientNet_B3_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b4":
            model = efficientnet.efficientnet_b4(EfficientNet_B4_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b5":
            model = efficientnet.efficientnet_b5(EfficientNet_B5_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b6":
            model = efficientnet.efficientnet_b6(EfficientNet_B6_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "efficientnet_b7":
            model = efficientnet.efficientnet_b7(EfficientNet_B7_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = None
            print(f"No EfficientNet model found with key {cfg.model_arch}")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch.startswith("vit_"):
        if cfg.model_arch == "vit_large_r50_s32_384":
            model = timm.create_model(cfg.model_arch, num_classes=cfg.num_classes, pretrained=True)
            return model
        elif cfg.model_arch == "vit_large_patch32_384":
            model = timm.create_model(cfg.model_arch, num_classes=cfg.num_classes, pretrained=True)
            return model
        else:
            print(f"NO fastai.timm model found with the key {cfg.model_arch}")

    elif cfg.model_arch.startswith("swin"):
        if cfg.model_arch == "swin_t":
            model = swin_t(weights = Swin_T_Weights.DEFAULT, progress =True)
            model.head = nn.Linear(model.head.in_features, cfg.num_classes)
            return model
        elif cfg.model_arch == "swin_s":
            model = swin_s(weights = Swin_S_Weights.DEFAULT, progress =True)
            model.head = nn.Linear(model.head.in_features, cfg.num_classes)
            return model
        elif cfg.model_arch == "swin_b":
            model = swin_b(weights = Swin_B_Weights.DEFAULT, progress =True)
            model.head = nn.Linear(model.head.in_features, cfg.num_classes)
            return model
        else:
            print(f"NO Swin Transformer model found with the key {cfg.model_arch}")






    return None


if __name__ == "__main__":
    from configs.config import cfg

    ml_model = get_model(cfg)
    print(ml_model)
    # summary(ml_model.cuda(), (3, 512, 512))
