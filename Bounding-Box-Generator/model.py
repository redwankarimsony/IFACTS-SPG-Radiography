import shutil
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def save_checkpoint(state, is_best, checkpoint_dir):
    # Source: https://medium.com/analytics-vidhya/saving-and-loading-your-model-to-resume-training-in-pytorch-cb687352fa61
    f_path = f"{checkpoint_dir}/checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        best_fpath = f"{checkpoint_dir}/best_model.pt"
        shutil.copyfile(f_path, best_fpath)


def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_object_detection_model(num_classes=2)
    print(model)
