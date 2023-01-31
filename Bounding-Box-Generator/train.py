import torch
import argparse
import json
from model import get_object_detection_model, save_checkpoint

from dataset import load_dataset
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from engine import  train_one_epoch, evaluate


# from torch.utils.data import DataLoader
# from pytorch_vision_library import utils


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Loading Dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def main(config):
    # to train on gpu if selected.
    device = torch.device(config["cuda_device"]) if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_object_detection_model(num_classes=2)
    # model = nn.DataParallel(model)
    model = model.to(device)


    # Loading the data
    dl_train, dl_valid, dl_test = load_dataset(config)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # training for 10 epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, dl_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluation = evaluate(model, dl_test, device=device)

        # print(evaluation.coco_eval.__dict__())

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint, True, "saved_models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", required=True)

    args = parser.parse_args()
    print(args)

    with open(args.config, "r") as f:
        config = f.read()
        config = json.loads(config)

    main(config=config)
