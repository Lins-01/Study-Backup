from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, CompareTest
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.GRU import GRUModel
from tee import StdoutTee
import pandas as pd
import pickle

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
parser.add_argument('--is_gpt', type=int, default=1)
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

if __name__ == '__main__':

    model_path = './checkpoints/1-17-GRU_sl1024_ll128_pl1024_dm768_nh4_el3_gl6_df768_ebtimeF_itr0_tgvol_170101010.tempf_epoch10_dataset_period_loop1_resampled_file222.csv'

    output_log = model_path + '/' + 'output2.log'
    with StdoutTee(output_log, mode="a", buff=1):

        device = torch.device('cuda:0')
        if args.model == 'PatchTST':
            model = PatchTST(args, device)
            model.to(device)
        elif args.model == 'DLinear':
            model = DLinear(args, device)
            model.to(device)
        elif args.model == 'GRU':
            model = GRUModel(args)
            model.to(device)
        else:
            model = GPT4TS(args, device)

        # 加载最好的模型

        best_model_path = model_path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))

        # 加载输入
        min_input = pd.read_csv('min_input.csv')

        pred = CompareTest(model, min_input, args, device)

        print("min_pred-------------------------------------min_true")
        # 将数据转换为 NumPy 数组
        pred_array = np.array(pred)

        # 移除 batch 维度
        pred_array = np.squeeze(pred_array, axis=0)

        # 确保 pred_array 现在是 2 维的 (1024, 1)
        print(pred_array.shape)  # 输出形状以确认

        # 加载scaler对象
        with open('scaler_test.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # 创建一个与scaler.scale_形状相同的全零数组，并将pred_array[:, 0]赋值给该数组的第一个元素
        data_expanded = np.zeros((pred_array.shape[0], scaler.scale_.shape[0]))
        data_expanded[:, 0] = pred_array[:, 0]  # [:,0] -> ：表示取所有行，0表示取第一列

        # 进行反归一化
        data_inverse = scaler.inverse_transform(data_expanded)

        # 将反归一化后的结果转换为Pandas DataFrame
        min_pred_df = pd.DataFrame(data_inverse[:, 0])

        # 将 DataFrame 保存为 CSV 文件
        min_pred_name = model_path + '/' + args.model + '_pred.csv'

        min_pred_df.to_csv(min_pred_name, index=False)

        print("Data saved to checkpoint's model_name_pred.csv")
