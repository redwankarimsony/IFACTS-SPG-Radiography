import torch.nn as nn
from torchsummary import summary
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models import ViT_B_32_Weights, vit_b_32
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.models import ViT_L_32_Weights, vit_l_32
from torchvision.models import ViT_H_14_Weights, vit_h_14





def get_model(model_name, num_classes=281):
    if model_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1, progress=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes, bias = True)
        return model
    elif model_name == "vit_b_32":
        model = vit_b_32(weights = ViT_B_32_Weights.IMAGENET1K_V1, progress = True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes, bias = True)
        return model
    elif model_name == "vit_l_16":
        model = vit_l_16(weights = ViT_L_16_Weights.IMAGENET1K_V1, progress = True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes, bias = True)
        return model
    elif model_name == "vit_l_32":
        model = vit_l_32(weights = ViT_L_32_Weights.IMAGENET1K_V1, progress = True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes, bias = True)
        return model
    elif model_name == "vit_h_14":
        model = vit_h_14(weights = ViT_H_16_Weights.IMAGENET1K_V1, progress = True)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes, bias = True)
        return model


if __name__ =="__main__":
    ml_model = get_model(model_name="vit_b_16")
    print(ml_model)



