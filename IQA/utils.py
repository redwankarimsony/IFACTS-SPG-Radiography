import torch

def save_model(model, path, optimizer, epoch):
    """_summary_

    Args:
        model (torch.nn.Module): _description_
        path (str): _description_
        optimizer (torch.optim): _description_
        epoch (int): _description_
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)