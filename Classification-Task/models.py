# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

import torch.nn as nn
from torchsummary import summary
from torchvision.models import densenet
from torchvision.models import efficientnet
from torchvision.models import resnet34, resnet50
from torchvision.models import ResNet34_Weights, ResNet50_Weights


def get_model(cfg):
    if cfg.model_arch == "resnet34":
        model = resnet34(weights=ResNet34_Weights.DEFAULT, progress=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    if cfg.model_arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    if cfg.model_arch == "densenet121":
        model = densenet.densenet121(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch == "densenet161":
        model = densenet.densenet161(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch == "densenet169":
        model = densenet.densenet169(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch == "efficientnet_b0":
        model = efficientnet.efficientnet_b0(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch == "efficientnet_b1":
        model = efficientnet.efficientnet_b1(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    elif cfg.model_arch == "efficientnet_b2":
        model = efficientnet.efficientnet_b2(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, cfg.num_classes, bias=True)
        return model

    return None


if __name__ == "__main__":
    from configs.config import cfg

    ml_model = get_model(cfg)
    print(ml_model)
    # summary(ml_model.cuda(), (3, 512, 512))
