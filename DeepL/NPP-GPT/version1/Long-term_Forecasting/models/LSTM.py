import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

class LSTMModel(nn.Module):
    def __init__(self, configs):
        super(LSTMModel, self).__init__()
        input_dim = 1  # Input feature dimension
        output_length = configs.pred_len
        n_layers = 1
        drop_prob = 0.2
        hidden_dim = 256

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_length)
        self.relu = nn.ReLU()

    def forward(self, x, itr):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        pre = out[:, -1, :]

        pre = self.relu(pre)  # Introduce non-linearity
        pre = self.fc(pre)
        pre = pre.unsqueeze(2)  # [batch_size, output_length] -> [batch_size, output_length, 1]
        return pre

