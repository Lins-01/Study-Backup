import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import torch
import argparse
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving = func(x)
            moving_mean.append(moving)
            sea = x - moving
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class Interactor_net(nn.Module):
    def __init__(self, in_dim=375, kernel=5, dropout=0.5, hidden_size=2):
        super(Interactor_net, self).__init__()

        self.in_dim = in_dim
        self.kernel_size = kernel
        self.dropout = dropout
        self.hidden_size = hidden_size

        if self.kernel_size % 2 == 0:
            pad_l = (self.kernel_size - 2) // 2 + 1
            pad_r = self.kernel_size // 2 + 1

        else:
            pad_l = (self.kernel_size - 1) // 2 + 1
            pad_r = (self.kernel_size - 1) // 2 + 1

        self.split = series_decomp_multi([3, 5, 7, 9])

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []

        modules_P += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(self.in_dim, int(self.in_dim * self.hidden_size), kernel_size=self.kernel_size, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(self.in_dim * self.hidden_size), self.in_dim, kernel_size=3, stride=1),
            nn.Tanh()
        ]

        modules_U += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(self.in_dim, int(self.in_dim * self.hidden_size), kernel_size=self.kernel_size, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(self.in_dim * self.hidden_size), self.in_dim, kernel_size=3, stride=1),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(self.in_dim, int(self.in_dim * self.hidden_size), kernel_size=self.kernel_size, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(self.in_dim * self.hidden_size), self.in_dim, kernel_size=3, stride=1),
            nn.Tanh()
        ]

        modules_psi += [
            nn.ReplicationPad1d((pad_l, pad_r)),
            nn.Conv1d(self.in_dim, int(self.in_dim * self.hidden_size), kernel_size=self.kernel_size, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv1d(int(self.in_dim * self.hidden_size), self.in_dim, kernel_size=3, stride=1),
            nn.Tanh()
        ]

        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

        self.con1 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)
        self.con2= nn.Conv1d(in_channels=128, out_channels=32, kernel_size=1)

    def forward(self, x):
        (x_even, x_odd) = self.split(x)

        x_even = x_even.permute(0, 2, 1)
        x_odd = x_odd.permute(0, 2, 1)

        d = x_odd.mul(torch.exp(self.phi(x_even)))
        c = x_even.mul(torch.exp(self.psi(x_odd)))

        x_even_update = c + self.U(d)
        x_odd_update = d - self.P(c)

        return self.con1(x_even_update.permute(0, 2, 1)) + self.con2(x_odd_update.permute(0, 2, 1))


if __name__ == '__main__':
    a = torch.randn([16, 128, 375])
    net = Interactor_net(in_dim=375, kernel=5, dropout=0.5, hidden_size=2)
    b = net(a)
    print(b.shape)