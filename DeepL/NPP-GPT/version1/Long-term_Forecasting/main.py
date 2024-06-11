from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.GRU import GRUModel
from models.LSTM import LSTMModel
from models.SCINet import SCINet
from tee import StdoutTee
import pandas as pd

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

        setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}_tg{}_epoch{}_dataset_{}'.format(args.model_id, args.seq_len,
                                                                                              args.label_len,
                                                                                              args.pred_len,
                                                                                              args.d_model,
                                                                                              args.n_heads,
                                                                                              args.e_layers,
                                                                                              args.gpt_layers,
                                                                                              args.d_ff, args.embed, ii,
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

            # 切分数据集，准备好三个数据集
            # data_provider里面会打印 train/val/test 个数
            train_data, train_loader = data_provider(args, 'train')
            vali_data, vali_loader = data_provider(args, 'val')
            test_data, test_loader = data_provider(args, 'test')

            # freq传入0后，这里不用管
            if args.freq != 'h':
                args.freq = SEASONALITY_MAP[test_data.freq]
                print("freq = {}".format(args.freq))

            device = torch.device('cuda:0')

            time_now = time.time()
            train_steps = len(train_loader)

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
            else:
                model = GPT4TS(args, device)
            # mse, mae = test(model, test_data, test_loader, args, device, ii)

            # model.parameters() 返回一个迭代器，迭代器中每个元素是model每层权重的值
            params = model.parameters()
            model_optim = torch.optim.Adam(params, lr=args.learning_rate)

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

            # 动态学习率的变化设定
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
            for epoch in range(args.train_epochs):

                iter_count = 0
                train_loss = []
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(device)

                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)

                    outputs = model(batch_x, ii)

                    outputs = outputs[:, -args.pred_len:, :]
                    batch_y = batch_y[:, -args.pred_len:, :].to(device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 1000 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    loss.backward()
                    model_optim.step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                train_loss = np.average(train_loss)
                vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
                # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
                # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
                #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

                if args.cos:
                    scheduler.step()
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(model_optim, epoch + 1, args)
                # 调用EarlyStopping中的__call__
                early_stopping(vali_loss, model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # 加载最好的模型
            best_model_path = path + '/' + 'checkpoint.pth'
            model.load_state_dict(torch.load(best_model_path))

            mse, mae, rmse, mape, min_pred, min_true, min_rmse, min_input = test(model, test_data, test_loader, args, device, ii)

            print("min_pred-------------------------------------min_true")
            # 将数据转换为 NumPy 数组
            min_pred_array = np.array(min_pred)
            min_true_array = np.array(min_true)
            min_input_array = np.array(min_input)

            # 将数据转换为 Pandas DataFrame
            min_pred_df = pd.DataFrame(min_pred_array, columns=['min_pred'])
            min_true_df = pd.DataFrame(min_true_array, columns=['min_true'])
            min_input_df = pd.DataFrame(min_input_array, columns=['min_input'])

            # 将 DataFrame 保存为 CSV 文件
            min_pred_name = path + '/' + 'min_pred.csv'
            min_true_name = path + '/' + 'min_true.csv'
            min_input_name = path + '/' + 'min_input.csv'

            min_true_df.to_csv(min_true_name, index=False)
            min_pred_df.to_csv(min_pred_name, index=False)
            min_input_df.to_csv(min_input_name, index=False)

            print("Data saved to checkpoint's min_pred.csv and min_true.csv and min_input.csv")

            print("------------------------------------")
            rmses.append(rmse)
            mapes.append(mape)
            mses.append(mse)
            maes.append(mae)

            rmses = np.array(rmses)
            mapes = np.array(mapes)
            mses = np.array(mses)
            maes = np.array(maes)
            # 多次迭代的平均值
            print("多次迭代才有意义的平均值!")
            print("rmse_mean = {:.4f}, rmse_std = {:.4f}".format(np.mean(rmses), np.std(rmses)))
            print("mape_mean = {:.4f}, mape_std = {:.4f}".format(np.mean(mapes), np.std(mapes)))
            print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
            print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
            plt.figure(figsize=(12, 6))
            label1 = args.model+'_pred' + '_min_rmse=' + str(min_rmse)
            plt.plot(min_pred, label=label1)
            plt.plot(min_true, label=args.target)
            plt.legend()  # 显示图例,即label
            res_name = './res/' + setting + '.png'
            plt.savefig(res_name)
            print('min_rmse:{:.4f}'.format(min_rmse))
            print("循环里面，最后一次是总的平均值！")

    # mses = np.array(mses)
    # maes = np.array(maes)
    # print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
    # print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
    print("运行到了这里！")
