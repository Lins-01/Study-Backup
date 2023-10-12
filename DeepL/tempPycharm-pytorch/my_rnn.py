import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.optim.lr_scheduler import MultiStepLR
from pylab import mpl
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import torch.optim.lr_scheduler as lr_scheduler
import warnings
import argparse

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--predicted_num", type=int, default=30, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=7, help="number of cpu threads to use during batch generation")
parser.add_argument("--learning_rate", type=int, default=0.1, help="learning rate")
parser.add_argument("--num_time_steps", type=int, default=16, help="num_time_steps")  # 训练时时间窗的步长
parser.add_argument("--input_size", type=int, default=2, help="input_size")  # 输入数据维度
parser.add_argument("--hidden_size", type=int, default=2, help="hidden_size")  # 隐含层维度
parser.add_argument("--output_size", type=int, default=2, help="output_size")  # 输出维度
parser.add_argument("--num_layers", type=int, default=5, help="num_layers")

mm = MinMaxScaler()

opt = parser.parse_args(args=[])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, total_length):
        # def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        # self.num = batchsize
        # self.test_num = test_batch_size
        self.rnn = nn.GRU(  # 改nn.GRU看看，lstm的更长记忆，但简化版lstm
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)

        self.linear1 = nn.Linear(total_length * hidden_size, 512)
        self.linear2 = nn.Linear(512, opt.output_size * 30)
        self.leak = nn.LeakyReLU()

    def forward(self, x, hiddern_prev):
        # print('x shape:', x.shape)
        # print('hiddern_prev shape:', hidden_prev.shape)
        out, hiddern_prev = self.rnn(x, hiddern_prev)
        # [b, seq, h]
        # print('out.shape[0]:', out.shape[0])
        out = out.reshape(out.shape[0], -1)
        out = self.leak(self.linear1(out))  # [seq,h] => [seq,3]
        out = self.linear2(out)  # [seq,h] => [seq,3]
        out = out.view(out.shape[0], 30, 2)

        # out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hiddern_prev


train_file_path = '1.txt'
trajectory_data_train, max_len = read_trajectory_data(train_file_path)

test_file_path = '1.txt'
trajectory_data_test, max_len1 = read_trajectory_data(test_file_path)

# 创建训练集和测试集的数据集实例
train_dataset = TrajectoryDataset(trajectory_data_train, mode='train')
test_dataset = TrajectoryDataset(trajectory_data_test, mode='test')
collate_fn = CollateFn(max_len)
# print('collate_fn:', collate_fn)
# 创建数据加载器
train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=True,
                               drop_last=True, num_workers=opt.n_cpu)
# train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
#                                drop_last=True, num_workers=opt.n_cpu)
test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=False,
                              drop_last=True,
                              num_workers=opt.n_cpu)
# test_loader = data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=True,
#                               num_workers=opt.n_cpu)

model = Net(opt.input_size, opt.hidden_size, opt.num_layers, max_len).to(device)
# model = Net(opt.input_size, opt.hidden_size, opt.num_layers).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = \
    lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def train(model, train_loader, epochs):
    # logger.info("***************Start training!***************\n")

    model.train()
    hidden_prev = torch.zeros(opt.num_layers, opt.batch_size, opt.hidden_size)

    epoch_loss = 0

    # for i,(inputs, labels) in enumerate(train_loader):
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader)):
        inputs, labels, hidden_prev = inputs.to(device), labels.to(device), hidden_prev.to(device)

        out, _ = model(inputs, hidden_prev)

        loss = criterion(out, labels)

        epoch_loss += loss.item()

        # 后向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg = epoch_loss / len(train_loader)
    print(f"Epoch: {epoch + 1}/{epochs}")
    print(f"training set Loss: {loss_avg}\n")

    # 按epoch模型保存
    # if epoch % 10 == 9:
    torch.save(model, r'rnn_result/rnn_epoch_%03d.pkl' % (epoch + 1))

    # 保存损失最低的模型
    # if loss_avg < temp:
    #     torch.save(model, r'rnn_result/lstm_best.pkl')
    #     temp = loss_avg


def print_tensor_list(tensor_list):
    string = ""
    # array = tensor.squeeze().cpu().numpy()
    for tensor in tensor_list:
        # 将tensor转换为numpy数组
        # array = tensor.numpy()
        # 格式化为坐标，并添加到string
        coordinate = f"{tensor[0]:.6f} {tensor[1]:.6f}"
        string += coordinate
        string += ", "

    # 删除尾部多余的comma and space
    string = string[:-2]
    return string
    # print(string)


def test(model, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        # running_loss = 0.0

        hidden_prev1 = torch.zeros(opt.num_layers, opt.batch_size, opt.hidden_size)
        for inputs, targets in test_loader:
            # inputs = inputs.unsqueeze(1).to(device)

            outputs, _ = model(inputs.to(device),
                               hidden_prev1.to(device))  # .unsqueeze(1).view(1, 1, opt.predicted_num, 2)
            last_elements = inputs[:, -1, :]
            last_elements = last_elements.unsqueeze(1)
            input_list = np.array(last_elements.cpu()).tolist()
            list_to_append = [tensor.cpu().numpy() for tensor in torch.split(outputs, 1, dim=0)]
            for i in range(len(input_list)):
                input_list[i] = np.concatenate([input_list[i], list_to_append[i].squeeze()], 0)

            # list_of_tensors = torch.split(outputs, 1, dim=0)
            for tensor_list in input_list:
                corrd_string = print_tensor_list(tensor_list)
                with open(r"rnn_result\pre_coords.txt", "a") as q:
                    q.write(
                        "|Epoch %d/%d| per_co:%s\r\n" % (epoch + 1, opt.n_epochs, corrd_string))
                    q.close()
                # print('\n')
            # array_list = [tensor.cpu().numpy() for tensor in list_of_tensors]
            # for idx, array in enumerate(array_list):
            #     print(f"Tensor {idx}:\n {array}")
            # outputs = torch.split(outputs, split_size_or_sections=1, dim=0)
            # outputs_coord = outputs.tolist()
            # predicted_coords_str = ",".join([f"{x:.9f} {y:.9f}" for x, y in array_list])
            # with open(r"rnn_result\pre_coords.txt", "a") as q:
            #     q.write("|Epoch %d/%d| |num=%d| per_co:%s\r\n" % (epoch+1, opt.n_epochs, num, predicted_coords_str))
            #     q.close()

            # input_list = np.array(inputs.squeeze(0).squeeze(0).cpu()).tolist()
            # coord = input_list[-1:]
            # outputs_cpu = np.array(outputs.squeeze(0).cpu()).tolist()
            # for sublist in outputs_cpu:
            #     coord.append(sublist)
            #
            # predicted_coords_str = ",".join([f"{x:.9f} {y:.9f}" for x, y in coord])
            # with open(r"rnn_result\pre_coords.txt", "a") as q:
            #     q.write("Epoch %d/%d| |num=%d| per_co:%s\r\n" % (epoch+1, opt.n_epochs, num, predicted_coords_str))
            #     q.close()
            # num += 1


if __name__ == '__main__':
    for epoch in range(opt.n_epochs):
        train(model, train_loader, epochs=opt.n_epochs)
        # if (epoch + 1) % 20 == 0:
        test(model, test_loader, epoch)
