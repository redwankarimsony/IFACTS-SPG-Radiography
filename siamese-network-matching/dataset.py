import os, glob
from itertools import combinations

from torch.utils.data import Dataset, DataLoader



class MatchingDataset(Dataset):
    def __init__(self, dataRoot:str = '../partial-match-data' , train:bool=False) -> None:
        super(MatchingDataset, self).__init__()
        all_pairs = glob.glob(f"{dataRoot}/*/*.jpg")
        all_combs = combinations(all_pairs, 2)        
        self.img_paths = []
        for pair in all_combs:
            if self.pair_validation(pair):
                self.img_paths.append(pair)

        # print(self.img_paths)

    def pair_validation(self, pair):
        first, second = pair[0].split(os.path.sep)[-1], pair[1].split(os.path.sep)[-1]
        orientFirst, orientSecond = first[13:15], second[13:15]
        return orientFirst==orientSecond

    def __len__(self):
        return len(self.img_paths)

    def getLabel(self, pair):
        first, second = pair[0].split(os.path.sep)[-1], pair[1].split(os.path.sep)[-1]
        return (first.split("_")[0] == second.split("_")[0])*1

    def __getitem__(self, idx) :
        return self.img_paths[idx], self.getLabel(self.img_paths[idx])





if __name__ == '__main__':
    ds = MatchingDataset()

    for i in range(len(ds)):
        print(ds[i])