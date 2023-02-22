import torch.nn as nn
import timm
import torch

from torchsummary import summary
from configs.config import cfg
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models import ViT_B_32_Weights, vit_b_32
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.models import ViT_L_32_Weights, vit_l_32
from torchvision.models import ViT_H_14_Weights, vit_h_14

def get_model(model_name, num_classes=281):
    if cfg.model_arch.startswith("vit_"):
        if cfg.model_arch == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "vit_b_32":
            model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "vit_l_16":
            model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "vit_l_32":
            model = vit_l_32(weights=ViT_L_32_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch  == "vit_h_14":
            model = vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = None
            print(f"No ResNet model found with key {cfg.model_arch}")
        model.heads.head = nn.Linear(model.heads.head.in_features, cfg.num_classes, bias=True)
        return model

def get_model_timm(cfg):
    # model = timm.create_model("vit_large_r50_s32_384", num_classes=281, pretrained=True)
    print(timm.list_models("vit_large*"))
    # print(model)
    # return model


if __name__ == "__main__":
    # ml_model = get_model(cfg)
    # print(ml_model)
    vit_model = get_model_timm(None)
    # x = torch.randn(1, 3, 384, 384)

    # output = vit_model(x)
    # print(output.shape)

