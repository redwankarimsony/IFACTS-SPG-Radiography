# Created by sonymd at 2/5/23

from torchvision.models import densenet
from torchvision.models import efficientnet
import torch.nn as nn


def get_model(cfg):
    if cfg["model_arch"] == "densenet121":
        model = densenet.densenet121(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg["num_classes"]
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg["model_arch"] == "densent161":
        model = densenet.densenet161(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg["num_classes"]
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg["model_arch"] == "densenet169":
        model = densenet.densenet169(pretrained=True, progress=True)
        in_features = model.classifier.in_features
        out_features = cfg["num_classes"]
        model.classifier = nn.Linear(in_features, out_features, bias=True)

    elif cfg["model_arch"] == "efficientnet_b0":
        model = efficientnet.efficientnet_b0(pretrained=True, progress=True)
        in_features = model.classifier[1].in_features
        out_features = cfg["num_classes"]
        model.classifier[1] = nn.Linear(in_features, out_features, bias=True)

    return model


if __name__ == "__main__":
    ml_model = get_model(cfg={"model_arch": "efficientnet_b0",
                              "num_classes": 280})
    print(ml_model)
