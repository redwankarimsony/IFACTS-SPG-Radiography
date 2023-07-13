from .dataset import ImageQualityDataset, transform
from .model import get_model
from torch.utils.data import DataLoader
import torch
from statistics import mean
import argparse
from .utils import save_model




def train(model, dl_train, dl_valid, optimizer, criterion, 
          epochs=10, 
          lr=0.001, 
          device='cuda:0' 
          if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    for epoch in range(epochs):
        best_val_loss =1e6
        model.train()
        train_losses = []
        for i, (images, labels) in enumerate(dl_train):
            
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_loss = mean(train_losses)
            # print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
        # print(f'Epoch Summary: {epoch + 1}/{epochs}, Loss: {train_loss:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            for i, (images, labels) in enumerate(dl_valid):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                # print(f'Epoch: {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')
        
        val_loss = mean(val_losses)
        print(f'Epoch Summary: {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # print(f'Epoch Summary: {epoch + 1}/{epochs}, ')

        # Save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, optimizer=optimizer, epoch=epoch, path= f'IQA/weights/epoch_{epoch+1}_val_loss_{val_loss:.4f}.pt')
            print(f'Model saved at epoch {epoch+1} with loss {val_loss:.4f}')

        # early stopping
        if val_loss > train_loss:
            print(f'Early Stopping at epoch {epoch+1}')
            break   
        

        


def load_datasets(cfg):
    # Loading the dataset 
    ds_train = ImageQualityDataset(data_path=cfg.data_path_train, transform=transform)
    ds_valid = ImageQualityDataset(data_path=cfg.data_path_valid, transform=transform)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)
    dl_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=cfg.shuffle, pin_memory=cfg.pin_memory, num_workers=cfg.num_workers)

    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    return ds_train, ds_valid, dl_train, dl_valid, device


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path_train', type=str, default='IQA/dummy_dataset/train')
    parser.add_argument('--data_path_valid', type=str, default='IQA/dummy_dataset/valid')   
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)  
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda:0')  
    cfg = parser.parse_args()

    return cfg

    

if __name__ == '__main__':
    # Loading the config   
    cfg = load_config()

    # Loading the dataset
    ds_train, ds_valid, dl_train, dl_valid, device = load_datasets(cfg)
    
    # Loading the model
    model = get_model()

    # Set the optimizer and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Start training
    train(model, dl_train, dl_valid, epochs=cfg.epochs, optimizer=optimizer, criterion=criterion, lr=cfg.lr, device=cfg.device)

        
