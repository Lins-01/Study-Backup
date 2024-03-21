import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler


class Datasets(Dataset):
    def __init__(self, path, in_seq_len=10, out_seq_len=3):
        self.x_train, self.y_train = [], []

        data_names = os.listdir(path)
        for data_name in data_names:
            data = pd.read_csv(os.path.join(path, data_name)).drop(['TrendTime', 'ProblemTime'], axis=1).values

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            time_stamp_1 = in_seq_len    # 输入序列长度
            time_stamp_2 = out_seq_len   # 输出序列长度

            # 切片
            for i in range(time_stamp_1, min(len(scaled_data) - time_stamp_2, 3000)):
                self.x_train.append(scaled_data[i - time_stamp_1:i])
                self.y_train.append(scaled_data[i: i + time_stamp_2])

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        return torch.FloatTensor(self.x_train[index]),  torch.FloatTensor(self.y_train[index])


if __name__ == '__main__':
    dataset = Datasets(path=r"./data", in_seq_len=128, out_seq_len=12)
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=12)

    k = 1
    for a, b in tqdm(loader):
        print(k)
        print(a.shape)
        print(b.shape)

        k = k + 1
