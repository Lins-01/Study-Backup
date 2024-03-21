import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

class Datasets(Dataset):
    # 初始化函数：读取数据、归一化、划分数据集
    def __init__(self, path, in_seq_len=10, out_seq_len=3):
        self.x_train, self.y_train = [], []


        data_names = os.listdir(path)
        for data_name in data_names:
            df = pd.read_csv(os.path.join(path, data_name)).drop(['TrendTime', 'ProblemTime'], axis=1).values

            # 要预测的列
            label_col = 'vol_170101010.tempf'
            # 总的输入特征
            data = df[[label_col, 'junc_170101014.mflowj', 'junc_170101024.mflowj', 'vol_340070000.tempf',
                       'vol_440150000.tempf']]
            print(data.head())
            print(data.shape)


            sc = MinMaxScaler(feature_range=(0, 1))
            label_sc = MinMaxScaler(feature_range=(0, 1))

            data_normalized = sc.fit_transform(data)


            # 单独保存要预测那一列的归一化参数 ，以备反归一化使用,因为这里对5维归一化，最后预测维度不一样，直接调用api出错
            print("df[label_col].shape: ", df[label_col].shape)
            # 因为归一化默认是多个特征，即二维，所以需要reshape(-1, 1)转为2D
            label_sc.fit(df[label_col].values.reshape(-1, 1))

            print(data_normalized.shape)
            print(data_normalized[0:5])





    def __len__(self):
        return len(self.x_train)

    # 滑动窗口
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