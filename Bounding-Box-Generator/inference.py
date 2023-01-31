import torch
import argparse
import json
import torchvision
from torchvision import transforms
from model import get_object_detection_model, load_checkpoint

from dataset import load_dataset

from dataset import ObjectDetectionDataset, get_transform


# the function takes the original prediction and the iou threshold.

def apply_nms(orig_prediction, iou_thresh=0.3):
    """

    Args:
        orig_prediction: original bounding box output from the model
        iou_thresh: intersection over union (iou) threshold

    Returns: Returns the filtered prediction which has iou_value greater than the given iou_threshold.

    """
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def torch_to_pil(img):
    """
    Args:
        img: a torch.tensor image [C x W x H]

    Returns: PIL image of the same size

    """
    return transforms.ToPILImage()(img).convert('RGB')


def main(config):
    dataset_test = ObjectDetectionDataset(config["test_dir"],
                                          config["width"],
                                          config["height"],
                                          transforms=get_transform(train=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(args)

    with open(args.config, "r") as f:
        config = f.read()
        config = json.loads(config)

    main(config=config)
