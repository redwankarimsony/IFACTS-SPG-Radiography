import imp
import os
from torch.utils.data import DataLoader

from dataset import TripletDataset

# Loading Dataset
ds_train = TripletDataset(dataDir="../partial-ten-more", split="train")
ds_valid = TripletDataset(dataDir="../partial-ten-more", split="valid")


dl_train = DataLoader(ds_train, batch_size=8, shuffle=False, num_workers=int(os.cpu_count()*0.75))
dl_valid = DataLoader(ds_valid, batch_size=8, shuffle=False, num_workers=int(os.cpu_count()*0.75))

