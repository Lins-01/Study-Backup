import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

class GRUWithAttentionModel(nn.Module):
    def __init__(self, configs):
        super(GRUWithAttentionModel, self).__init__()
        input_dim = 1
        output_length = configs.pred_len
        n_layers = 1
        drop_prob = 0.2
        hidden_dim = 256

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_length)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)

        attn_weights = self.attention(out).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)

        pre = self.relu(context_vector)
        pre = self.fc(pre)
        pre = pre.unsqueeze(2)

        return pre

# Define configuration class for parameters
class Configs:
    def __init__(self, pred_len):
        self.pred_len = pred_len

# Example usage
configs = Configs(pred_len=96)

# Define model
model = GRUWithAttentionModel(configs)

# Example forward pass
x = torch.randn(256, 10, 1)
output = model(x)

print(output.shape)  # Should be [256, 96, 1]
