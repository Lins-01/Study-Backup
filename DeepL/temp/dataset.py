import os
import pandas as pd
from torch.utils.data import Dataset


class Datasets(Dataset):
    def __init__(self, path):
        # 需要的数据
        self.name = pd.read_csv(path)

    def __len__(self):
        return len(self.name)
        # return self.name.shape[0]-14

    def __getitem__(self, index):
        print(index)
        print(len(self.name))
        i = index
        return self.name[i:i+10], self.name[i+10:i+13]


if __name__ == '__main__':
    i = 0
    dataset = Datasets(r"E:\Document\CodeSpace\Study\DeepL\temp\POWER0.3-0.4_LOOP1.csv")
    for a, b in dataset:
        print(a.shape)
        print(b.shape)

        i += 1

    print(i)
