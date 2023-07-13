from torchvision.models import resnet50,resnet18,ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights
import torch



def get_model(model_code="mobilenet_v2", weights="imagenet", num_classes=4):
    if model_code == "mobilenet_v2" and weights=="imagenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
        return model
    elif model_code == "resnet50" and weights=="imagenet":
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_code == "resnet18" and weights=="imagenet":
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise NotImplementedError("Model not implemented")
        return None



