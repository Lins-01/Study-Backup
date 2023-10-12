import torch
import os
import torch.nn as nn
import torch.optim as optim
# import dataset
import logging
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--predicted_num", type=int, default=30, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--learning_rate", type=int, default=0.0001, help="learning rate")

opt = parser.parse_args(args=[])

write = SummaryWriter("runs/LSTM")

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("LSTM.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取轨迹数据
def read_trajectory_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        tensor_list = []
        max_len = 0
        for line in lines:
            points = line.split(", ")
            coords = []
            for point in points:
                longitude, latitude = map(float, point.split(" "))
                coords.append([longitude, latitude])
            coords_tensor = torch.tensor(coords)
            tensor_list.append(coords_tensor)
            if len(coords) > max_len:  # Update max length
                max_len = len(coords)
    return tensor_list, max_len


# 创建自定义数据集类
class TrajectoryDataset(data.Dataset):
    def __init__(self, trajectory_data, mode):
        self.trajectory_data = trajectory_data
        self.mode = mode

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, idx):
        trajectory = torch.Tensor(self.trajectory_data[idx])
        if self.mode == 'train':
            inputs = trajectory[:-30]
            targets = trajectory[-30:]
            return inputs, targets
        elif self.mode == 'test':
            inputs = trajectory[:]
            targets = torch.from_numpy(np.zeros((30, 2)))
            return inputs, targets


# def collate_fn(batch, sequence_length):
#     inputs_batch, targets_batch = [], []
#     for item in batch:
#         inputs, targets = item
#         inputs_batch.append(inputs)
#         if targets is not None:
#             targets_batch.append(targets)
#     inputs_padded = pad_sequence(inputs_batch, batch_first=True, padding_value=0, total_length=sequence_length).flip(
#         dims=[1]) # Padding and flipping inputs
#     if len(targets_batch) > 0:  # If targets exist
#         targets_padded = pad_sequence(targets_batch, batch_first=True, padding_value=0,
#                                       total_length=sequence_length).flip(dims=[1]) # Padding and flipping targets
#         return inputs_padded, targets_padded
#     else:
#         return inputs_padded

class CollateFn:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def __call__(self, batch):
        inputs_batch, targets_batch = [], []
        for item in batch:
            inputs, targets = item
            inputs_batch.append(inputs)
            if targets is not None:
                targets_batch.append(targets)
        inputs_padded = pad_sequence(inputs_batch, batch_first=True).flip(dims=[1])
        inputs_padded = torch.cat(
            [torch.zeros(inputs_padded.shape[0], self.sequence_length - inputs_padded.shape[1], inputs_padded.shape[2]),
             inputs_padded], dim=1)
        if len(targets_batch) > 0:
            targets_padded = pad_sequence(targets_batch, batch_first=True).flip(dims=[1])
        return inputs_padded, targets_padded


class LSTMModel(nn.Module):
    def __init__(self, input_size, in_seq_len, hidden_size=200,  out_seq_len=30, lstm_num_layers=5):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_seq_len = out_seq_len
        self.in_seq_len = in_seq_len
        self.lstm_num_layers = lstm_num_layers

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.fc = nn.Linear(self.in_seq_len * self.hidden_size, self.out_seq_len * self.input_size)

    def forward(self, x):
        # a = x.shape[1]
        out, _ = self.lstm(x)
        batch_size = out.shape[0]
        out = out.contiguous().view(batch_size, -1)
        # out .view(batch_size, -1)
        # fc = nn.Linear(a * 200, 60)
        out = self.fc(out.to(device)).to(device)
        # out = fc(out)
        out = out.view(batch_size, self.out_seq_len, -1)
        return out

# if __name__ == '__main__':
#     b =torch.randn([16,500,2])
#     model = LSTMModel(input_size=2)
#     y=model(b)
#     print(y.shape)

# 示例数据
train_file_path = 'new.txt'
trajectory_data_train, max_len = read_trajectory_data(train_file_path)

test_file_path = 'new.txt'
trajectory_data_test, max_len1 = read_trajectory_data(test_file_path)

# 创建训练集和测试集的数据集实例
train_dataset = TrajectoryDataset(trajectory_data_train, mode='train')
test_dataset = TrajectoryDataset(trajectory_data_test, mode='test')
collate_fn = CollateFn(max_len)
# 创建数据加载器
train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True,
                               drop_last=True, num_workers=opt.n_cpu)
test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=False, drop_last=True,
                              num_workers=opt.n_cpu)

model = LSTMModel(input_size=2, in_seq_len=max_len).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def train(model, train_loader, epochs):
    logger.info("***************Start training!***************\n")

    # temp = 10000
    model.train()

    # for epoch in range(opt.n_epochs):

    epoch_loss = 0

    # for i, (inputs, labels) in enumerate(train_loader):
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        out = model(inputs)

        loss = criterion(out, labels)

        epoch_loss += loss.item()

        # 后向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg = epoch_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}/{epochs}")
    print(f"training set Loss: {loss_avg}\n")

    # 可视化模型, 第一轮时候遍历一次
    # if epoch == 0:
    #     write.add_graph(model, inputs)

    # 按epoch模型保存
    # if epoch % 10 == 9:
    torch.save(model, 'lstm_result/epoch_%03d.pth' % (epoch + 1))

    # # 保存损失最低的模型
    # if loss_avg < temp:
    #     torch.save(model, './lstm_result/best.pth')


def print_tensor_list(tensor_list):
    string = ""
    # array = tensor.squeeze().cpu().numpy()
    for tensor in tensor_list:

        coordinate = f"{tensor[0]:.6f} {tensor[1]:.6f}"
        string += coordinate
        string += ", "

    # 删除尾部多余的comma and space
    string = string[:-2]
    return string


def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        # running_loss = 0.0
        num = 1
        for inputs, _ in test_loader:
            inputs = inputs.unsqueeze(1).to(device)
            # targets = targets.unsqueeze(1).to(device)

            # outputs = model(inputs)
            outputs = model(inputs).unsqueeze(1).view(1, 1, opt.predicted_num, 2)
            outputs = outputs.squeeze(1)

            # loss = criterion(outputs, targets)
            # running_loss += loss.item() * inputs.size(0)

            input_list = np.array(inputs.squeeze(0).squeeze(0).cpu()).tolist()
            coord = input_list[-1:]
            outputs_cpu = np.array(outputs.squeeze(0).cpu()).tolist()
            for sublist in outputs_cpu:
                coord.append(sublist)

            predicted_coords_str = ",".join([f"{x:.9f} {y:.9f}" for x, y in coord])
            with open(r"lstm_result/pre_coords.txt", "a") as q:
                q.write("Epoch %d/%d| |num=%d| per_co:%s\r\n" % (epoch+1, opt.n_epochs, num, predicted_coords_str))
                q.close()
            num += 1



if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        train(model, train_loader, epochs=opt.n_epochs)
        if (epoch + 1) % 2 == 0:
            outputs_cpu = test(model, test_loader)
            # torch.save(model,
            #            r"D:\wh_ww\code\Kalman_test\cnn_cpu_%d.pkl" % (epoch + 1))