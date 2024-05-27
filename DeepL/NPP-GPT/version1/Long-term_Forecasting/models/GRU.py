import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

class GRUModel(nn.Module):
    def __init__(self, input_length, hidden_dim, output_length, n_layers, drop_prob=0.2):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        self.gru = nn.GRU(input_length, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device) # 不传入，默认为0
        # input(batch_size, seq_len, input_length)->x

        # 一次预测多步
        # output(batch_size, seq_len, num_directions * hidden_size)
        out, _ = self.gru(x)
        # batch和特征维度全取，seq_len维度取最后一个，即只取最后一个时间步的输出
        # out(batch_size, seq_len, num_directions * hidden_size)->out(batch_size, num_directions * hidden_size)
        pre = out[:, -1, :]

        # 全连接层把out维度从(batch_size, 1, num_directions * hidden_size)变成(batch_size, 1, output_length)
        pre = self.relu(pre)  # 引入非线性
        pre = self.fc(pre)
        # 转换维度为 [Batch, Output length, Channel]
        pre = pre.permute(0, 2, 1)
        return pre