# config/config.py
# Created by sonymd at 2/5/23
# Mail: sonymd@msu.edu
# GitHub: www.github.com/redwankarimsony
# Graduate Researcher, iPRoBe Lab, CSE, MSU

import torch.nn as nn
from torchsummary import summary
from torchvision.models import densenet
from torchvision.models import efficientnet


def get_model(cfg):
    model = None
    if cfg.model_arch == "densenet121":
        model = densenet.densenet121(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg.num_classes
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg.model_arch == "densent161":
        model = densenet.densenet161(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg.num_classes
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg.model_arch == "densenet169":
        model = densenet.densenet169(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg.num_classes
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg.model_arch == "efficientnet_b0":
        model = efficientnet.efficientnet_b0(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        out_features = cfg.num_classes
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)

    elif cfg.model_arch == "efficientnet_b1":
        model = efficientnet.efficientnet_b1(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        out_features = cfg.num_classes
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)

    elif cfg.model_arch == "efficientnet_b2":
        model = efficientnet.efficientnet_b2(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        out_features = cfg.num_classes
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)

    return model


if __name__ == "__main__":
    ml_model = get_model(cfg={"model_arch": "efficientnet_b2",
                              "num_classes": 280})
    summary(ml_model.cuda(), (3, 224, 224))
