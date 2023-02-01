import torch
import argparse
import json
import os
import torchvision
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from model import get_object_detection_model, load_checkpoint
from torch.utils.data import DataLoader

from dataset import load_dataset, coco_2_yolo

from dataset import ObjectDetectionInferenceDataset, get_transform, plot_img_bbox


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





def main(config):
    # Load the model
    # to train on gpu if selected.
    device = torch.device(config["cuda_device"]) if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_object_detection_model(num_classes=2)
    # model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint = torch.load(f"saved_models/best_model.pt")
    model.load_state_dict(checkpoint['state_dict'])

    ds_infer = ObjectDetectionInferenceDataset(config["infer_dir"],
                                               config["width"],
                                               config["height"],
                                               transforms=A.Compose([ToTensorV2(p=1.0)]))

    dl_infer = DataLoader(ds_infer,
                          num_workers=config["num_workers"],
                          shuffle=False,
                          pin_memory=config["pin_memory"])

    model.eval()

    with torch.no_grad():
        for epoch, (imgs, img_paths) in enumerate(dl_infer):
            predictions = model(imgs.to(device))
            for prediction, img_path in zip(predictions, img_paths):
                prediction = apply_nms(prediction, iou_thresh=0.7)
                if len(prediction["boxes"]):
                    label_file_name = img_path.split("/")[-1].replace(".png", ".txt")
                    with open(os.path.join(config["infer_save_dir"], label_file_name ), "w") as fp:
                        [label, x, y, w, h] = coco_2_yolo(prediction)
                        print(f"{label-1} {x:0.4f} {y:0.4f} {w:0.4f} {h:0.4f}", file=fp)
                else:
                    print(f"No prediction on {img_path}")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Script")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(args)

    with open(args.config, "r") as f:
        config = f.read()
        config = json.loads(config)

    main(config=config)
