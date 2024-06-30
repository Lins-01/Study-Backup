from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.GRU import GRUModel
from models.LSTM import LSTMModel
from models.SCINet import SCINet
from models.BERT import BERT4TS
from models.RoBERTaLoRA import RoBERTa4TS
from models.GPTLora import GPTLora


from torch.utils.data import DataLoader, Dataset
from tee import StdoutTee
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--bert_layers', type=int, default=3)
parser.add_argument('--roberta_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--is_bert', type=int, default=1)
parser.add_argument('--is_roberta', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)

args = parser.parse_args()


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, mask_prob=0.15):
        self.data = data
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.mask_len = int(seq_len * mask_prob)
        self.mask_token_id = -1  # Using -1 as the mask token ID for simplicity,归一化之后没有这么大的负数


        # 添加归一化之后，loss降了很多，从几万降到几千，第二次就一百，第三次十位数，第四次就个位数。
        # 归一化数据
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()  # 将数据归一化为 均值0，方差1，即在[-1, 1]之间

    def __len__(self):
        return len(self.data) - self.seq_len

    # def __getitem__(self, idx):
    #     seq = self.data[idx: idx + self.seq_len].copy().astype(np.float32)  # Ensure the sequence is a NumPy array
    #
    #
    #     # 这里args.seq_len 要和 args.pre_len一致
    #     labels = seq.copy()
    #     mask = np.random.rand(self.seq_len) < self.mask_prob
    #     seq[mask] = self.mask_token_id  # Mask some tokens
    #
    #     # # 将 Pandas Series 转换为 NumPy 数组，然后再转换为 PyTorch 张量
    #     # seq = seq.to_numpy()
    #     # labels = labels.to_numpy()
    #
    #     return torch.tensor(seq, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        seq = self.data[idx: idx + self.seq_len].copy().astype(np.float32)  # Ensure the sequence is a NumPy array

        labels = seq.copy()

        # 随机生成掩码的起始位置和长度
        start = int(np.random.randint(0, self.seq_len - self.mask_len + 1))
        end = int(start + self.mask_len)

        # 掩码序列中[start, end)的部分
        seq[start:end] = self.mask_token_id

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)




SEASONALITY_MAP = {
    "minutely": 1440,
    "10_minutes": 144,
    "half_hourly": 48,
    "hourly": 24,
    "daily": 7,
    "weekly": 1,
    "monthly": 12,
    "quarterly": 4,
    "yearly": 1
}

mses = []
maes = []
rmses = []
mapes = []

if __name__ == '__main__':
    for ii in range(args.itr):

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}_tg{}_epoch{}_dataset_{}'.format(args.model_id,
                                                                                                         args.seq_len,
                                                                                                         args.label_len,
                                                                                                         args.pred_len,
                                                                                                         args.d_model,
                                                                                                         args.n_heads,
                                                                                                         args.e_layers,
                                                                                                         args.gpt_layers,
                                                                                                         args.d_ff,
                                                                                                         args.embed, ii,
                                                                                                         args.target,
                                                                                                         args.train_epochs,
                                                                                                         args.data_path)

        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 与best model放一起
        output_log = path + '/' + 'output.log'
        with StdoutTee(output_log, mode="a", buff=1):
            # 传入的freq为0就默认是小时为单位，我们这里就用0好了
            if args.freq == 0:
                args.freq = 'h'

            # Prepare dataset and dataloader
            data = pd.read_csv(os.path.join(args.root_path,
                                          args.data_path))
            data = data[args.target]
            num_train = int(len(data) * 0.8)
            # 测试集不给看。不泄露。
            data = data[:num_train]
            dataset = TimeSeriesDataset(data, seq_len=args.seq_len, mask_prob=0.15)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)



            device = torch.device('cuda:0')

            time_now = time.time()


            if args.model == 'PatchTST':
                model = PatchTST(args, device)
                model.to(device)
            elif args.model == 'DLinear':
                model = DLinear(args, device)
                model.to(device)
            elif args.model == 'GRU':
                model = GRUModel(args)
                model.to(device)
            elif args.model == 'LSTM':
                model = LSTMModel(args)
                model.to(device)
            elif args.model == 'SCINet':
                model = SCINet(args)
                model.to(device)
            elif args.model == 'BERT4TS':
                model = BERT4TS(args, device)
            elif args.model == 'RoBERTa4TS':
                model = RoBERTa4TS(args, device)
            elif args.model == 'GPTLora':
                model = GPTLora(args, device)
            else:
                model = GPT4TS(args, device)
            # mse, mae = test(model, test_data, test_loader, args, device, ii)

            # # 尝试加载已保存的模型权重
            # best_model_path = os.path.join(path, 'checkpoint.pth')
            # if os.path.exists(best_model_path):
            #     model.load_state_dict(torch.load(best_model_path))
            #     print(f"Loaded model weights from {best_model_path}")
            # else:
            #     print("加载的模型路径错误")

            # 打印加载后的模型权重,确认生效的权重。
            print("Loaded model weights (first few layers):================================")
            for name, param in list(model.named_parameters())[:5]:
                print(f"{name}: {param.data[:2]}")

            # model.parameters() 返回一个迭代器，迭代器中每个元素是model每层权重的值
            params = model.parameters()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1,threshold=5e-3, threshold_mode='abs',verbose=True, min_lr=1e-6)


            # 模型参数打印
            print("Trainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.shape}")
            print("\nFrozen parameters:")
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(f"{name}: {param.shape}")

            # EarlyStopping里有保存模型torch.save
            early_stopping = EarlyStopping(patience=args.patience, verbose=True)
            if args.loss_func == 'mse':
                criterion = nn.MSELoss()
            elif args.loss_func == 'smape':
                class SMAPE(nn.Module):
                    def __init__(self):
                        super(SMAPE, self).__init__()

                    def forward(self, pred, true):
                        return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))


                criterion = SMAPE()

            train_steps = len(dataloader)
            time_now = time.time()
            # Train the model
            model.train()
            for epoch in range(args.train_epochs):
                iter_count = 0
                train_loss = []
                epoch_time = time.time()
                for i,(inputs, labels) in tqdm(enumerate(dataloader)):
                    iter_count += 1
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    label_test = labels[1,:]
                    optimizer.zero_grad()
                    # 将inputs添加一个维度
                    inputs = inputs.unsqueeze(2)
                    outputs = model(inputs, 0)
                    # 将outputs减少一个维度
                    outputs = outputs.squeeze(2)
                    loss = criterion(outputs, labels)
                    train_loss.append(loss.item())

                    if (i + 1) % 1000 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    optimizer.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(
                    epoch + 1, train_steps, train_loss))
                scheduler.step(train_loss)
                print("lr = {:.10f}".format(optimizer.param_groups[0]['lr']))


            best_model_path = path + '/' + 'checkpoint.pth'
            torch.save(model.state_dict(), best_model_path)

            # 画出label和预测值的图

            # Visualize the outputs and labels
            model.eval()
            with torch.no_grad():
                val_inputs, val_labels = next(iter(dataloader))
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_inputs = val_inputs.unsqueeze(2)
                val_outputs = model(val_inputs, 0)
                val_outputs = val_outputs.squeeze(2)

            # Detach and move to CPU for plotting
            val_inputs = val_inputs.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()
            val_outputs = val_outputs.detach().cpu().numpy()

            idx = 0
            for i in range(10):
                idx = i * 25
                plt.figure(figsize=(12, 6))
                plt.plot(val_outputs[idx, :], label='Output')
                plt.plot(val_labels[idx, :], label='Label')
                plt.plot(val_inputs[idx, :, :], label='Input')
                plt.legend()
                res_name = './res/' + setting +str(idx)+ '.png'
                plt.show()
                plt.savefig(res_name)

