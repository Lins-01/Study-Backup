import torch
import os
import torch.nn as nn
import torch.optim as optim
import dataset
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from SCINet import SCINet
from test import Interactor_net

write = SummaryWriter("runs/LSTM")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("LSTM.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# class LSTMModel(nn.Module):
#     def __init__(self, input_size=375, hidden_size=1000, in_seq_len=10, out_seq_len=3, lstm_num_layers=5):
#         super(LSTMModel, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.in_seq_len = in_seq_len
#         self.out_seq_len = out_seq_len
#         self.lstm_num_layers = lstm_num_layers
#
#         self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.lstm_num_layers,
#                             batch_first=True)
#         self.fc = nn.Linear(self.in_seq_len * self.hidden_size, self.out_seq_len * self.input_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         batch_size = out.shape[0]
#         out = out.contiguous().view(batch_size, -1)
#         out = self.fc(out)
#         out = out.view(batch_size, self.out_seq_len, -1)
#         return out


class Trainer:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.net = SCINet(input_len=128, output_len=12, input_dim=375, hid_size=1, num_stacks=1, num_levels=2,
        #                   concat_len=0, groups=1, kernel=3, dropout=0.5, single_step_output_One=0, positionalE=True,
        #                   modified=True).to(self.device)

        self.net = Interactor_net(in_dim=375, kernel=5, dropout=0.5, hidden_size=2).to(self.device)

        data = dataset.Datasets(path=self.data_path, in_seq_len=128, out_seq_len=32)
        self.loader = DataLoader(data, batch_size=64, shuffle=True, num_workers=12)

        self.lr = 0.001
        self.opt = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)

        self.loss_func = nn.MSELoss()

        # if os.path.exists(self.model_path):
        #     self.net.load_state_dict(torch.load(self.model_path))
        #     print(f"Successful Loaded {self.model_path}!\n")
        # else:
        #     print("No model exist!\n")

    # 训练
    def train(self, epochs):

        logger.info("***************Start training!***************\n")

        temp = 10000

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0

            for inputs, labels in tqdm(self.loader, desc=f"Epoch {epoch+1}/{epochs}", total=len(self.loader)):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                out = self.net(inputs)

                loss = self.loss_func(out, labels)

                epoch_loss += loss.item()

                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            loss_avg = epoch_loss / len(self.loader)
            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"training set Loss: {loss_avg}\n")

            # 可视化模型, 第一轮时候遍历一次
            if epoch == 0:
                write.add_graph(self.net, inputs)

            # 按epoch模型保存
            if epoch % 10 == 9:
                torch.save(self.net.state_dict(), r'./model/epoch_%03d.pth' % (epoch + 1))

            # 保存损失最低的模型
            if loss_avg < temp:
                torch.save(self.net.state_dict(), r'./model/best.pth')
                temp = loss_avg

            logger.info(f"{epoch + 1}/{epochs}")
            logger.info(f"Lr: {self.opt.param_groups[0]['lr']}")
            logger.info(f"training set Loss: {loss_avg}\n")

            write.add_scalar('lr', self.opt.param_groups[0]['lr'], global_step=epoch + 1)
            write.add_scalar('training set Loss', loss_avg, global_step=epoch + 1)

        write.close()

        logger.info("***************End!***************")


if __name__ == '__main__':
    t = Trainer(data_path=r"./data", model_path=r"./model/best.pth")
    t.train(60)
