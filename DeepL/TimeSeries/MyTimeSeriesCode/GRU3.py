import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time


def create_dataloader(data_normalized, window_size, pred_len, batch_size):
    xs = []
    ys = []
    # Create the sequence data
    for i in range(len(data_normalized) - window_size - pred_len+1):
        x = data_normalized[i:i + window_size]
        y = data_normalized[i + window_size:i + window_size + pred_len, 0]  # 只预测第一个变量
        xs.append(x)
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    print("xs shape: ", xs.shape)
    print("ys shape: ", ys.shape)
    # 查看xy和ys的具体数据
    print("xs: ", xs[0])
    print("ys: ", ys[0])
    # Split into train and test
    train_test_split = 0.8
    split_idx = int(len(xs) * 0.8)

    train_x, test_x = xs[:split_idx], xs[split_idx:]
    train_y, test_y = ys[:split_idx], ys[split_idx:]
    print("train_x shape: ", train_x.shape)
    print("train_y shape: ", train_y.shape)
    print("test_x shape: ", test_x.shape)
    print("test_y shape: ", test_y.shape)
    # 预测值，每次滑动pre_len长度截取
    # 计算可以滑动的次数，每次滑动步长为pred_len
    num_slides = len(test_x) // pred_len
    print("num_slides: ", num_slides)

    # 初始化两个空列表来存储测试输入和测试输出
    test_x_new = []
    test_y_new = []


    # 使用循环来每间隔pred_len取一次，达到每次滑动pred_len长度的效果
    for i in range(num_slides):
        # 每次滑动pred_len长度截取
        # 计算当前滑动窗口的起始索引
        start_idx = i * pred_len
        # 计算当前滑动窗口的结束索引
        end_idx = start_idx + pred_len
        # 从test_x中截取当前滑动窗口的输入
        x_new = test_x[start_idx]
        # 从test_y中截取当前滑动窗口的预测值
        y_new = test_y[start_idx]
        # 将当前滑动窗口的输入和预测值都加入
        test_x_new.append(x_new)
        test_y_new.append(y_new)
    print("test_x_new shape: ", np.array(test_x_new).shape)
    print("test_y_new shape: ", np.array(test_y_new).shape)
    # 转为np.ndarray
    test_x_new = np.array(test_x_new)
    test_y_new = np.array(test_y_new)

    # Create tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data = TensorDataset(torch.from_numpy(test_x_new), torch.from_numpy(test_y_new))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size,
                              drop_last=True)  # drop_last=True表示如果最后一个batch_size不足，就丢弃
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)
    print("train_loader shape: ", len(train_loader))
    print("test_loader len: ", len(test_loader))

    return train_loader, test_loader


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initialize hidden state with zeros
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device) # 不传入，默认为0
        # input(batch_size, seq_len, input_size)->x

        # 一次预测多步
        # output(batch_size, seq_len, num_directions * hidden_size)
        out, _ = self.gru(x)
        # batch和特征维度全取，seq_len维度取最后一个，即只取最后一个时间步的输出
        # out(batch_size, seq_len, num_directions * hidden_size)->out(batch_size, num_directions * hidden_size)
        pre = out[:, -1, :]

        # 全连接层把out维度从(batch_size, 1, num_directions * hidden_size)变成(batch_size, 1, output_dim)
        pre = self.relu(pre)  # 引入非线性
        pre = self.fc(pre)
        return pre


def train(model, train_loader, input_dim, hidden_dim, output_dim, n_layers, epochs, lr, device):
    # We'll also set the model to the device that we defined earlier (default is CPU)
    model.to(device)
    # Define hyperparameters
    # Define Loss, Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    print("Starting Training of GRU model")
    epoch_times = []
    counter = 0
    # Train the model
    for epoch in range(epochs):
        start_time = time.process_time()
        avg_loss = 0.0

        for i, (x, y) in enumerate(train_loader):

            # Clear stored gradient
            model.zero_grad()

            # Make predictions, calculate loss, perform backprop
            # .to(device)不会就地修改x和y，而是返回一个新的张量，所以要用x = x.to(device)
            # 只用x.to(device)会报错
            # RuntimeError: Input and parameter tensors are not at the same device, found input tensor at cpu and parameter tensor at cuda:0
            x = x.to(device).float()
            y = y.to(device).float()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            counter += 1
            if counter % 100 == 0:
                f"Epoch {epoch} - Step: {counter}/{len(train_loader)} - Average Loss for Epoch: {avg_loss / counter}"
        current_time = time.process_time()

        print(
            f"Epoch {epoch}/{epochs} Done, Total Loss: {avg_loss / len(train_loader)}"
        )
        print(f"Time Elapsed for Epoch: {current_time - start_time} seconds")
        epoch_times.append(current_time - start_time)

    print(f"Total Training Time: {sum(epoch_times)} seconds")

    model_name = f"./models/model_{output_dim}.pth"
    torch.save(model.state_dict(), model_name)
    return model


def sMAPE(outputs, targets):
    sMAPE = (
            100
            / len(targets)
            * np.sum(np.abs(outputs - targets) / (np.abs(outputs + targets)) / 2)
    )
    return sMAPE


if __name__ == '__main__':
    dataset_path = 'fw80_pow90_loop1.csv'
    window_size = 90
    pred_len = 10
    batch_size = 64

    input_dim = 5  # 选了5个特征
    hidden_dim = 256
    n_layers = 2
    epochs = 100
    lr = 0.001
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("No GPU!!!!!!!!!!!!!!!!!!!!!")

    # Load the CSV file
    df = pd.read_csv(dataset_path)
    label_col = 'vol_170101010.tempf'
    data = df[[label_col, 'junc_170101014.mflowj', 'junc_170101024.mflowj', 'vol_340070000.tempf',
               'vol_440150000.tempf']]
    print(data.head())
    print(data.shape)
    sc = MinMaxScaler(feature_range=(0, 1))
    label_sc = MinMaxScaler(feature_range=(0, 1))

    data_normalized = sc.fit_transform(data)
    print(data_normalized.shape)
    print(data_normalized[0:5])

    # 单独保存要预测那一列的归一化参数 ,因为这里对5维归一化，最后预测维度不一样，直接调用api出错
    print("df[label_col].shape: ", df[label_col].shape)
    # 因为归一化默认是多个特征，即二维，所以需要reshape(-1, 1)转为2D
    label_sc.fit(df[label_col].values.reshape(-1, 1))

    train_loader, test_loader = create_dataloader(data_normalized, window_size, pred_len, batch_size)

    model = GRUModel(input_dim, hidden_dim, pred_len, n_layers)

    # model = train(model,train_loader, input_dim, hidden_dim, pred_len, n_layers, epochs, lr, device)

    # Evaluate the Model
    model.load_state_dict(torch.load(f"./models/model_{pred_len}.pth"))
    model.to(device)
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device).float()
            y = y.to(device).float()
            y_pred = model(x)
            predictions.append(y_pred.cpu().numpy())
            actuals.append(y.cpu().numpy())

    # 转为numpy
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    print("predictions shape: ", predictions.shape)
    print("actuals shape: ", actuals.shape)

    # 将三维数组转换为二维数组
    predictions = predictions.reshape(-1, pred_len)
    actuals = actuals.reshape(-1, pred_len)

    print("predictions shape: ", predictions.shape)
    print("actuals shape: ", actuals.shape)
    predictions_denorm = label_sc.inverse_transform(predictions)
    actuals_denorm = label_sc.inverse_transform(actuals)

    print("predictions_denorm shape: ", predictions_denorm.shape)
    print("actuals_denorm shape: ", actuals_denorm.shape)
    concatenated_outputs = predictions_denorm.reshape(-1, 1)
    concatenated_actuals = actuals_denorm.reshape(-1, 1)
    print("concatenated_outputs shape: ", concatenated_outputs.shape)
    print("concatenated_actuals shape: ", concatenated_actuals.shape)
    # Calculate sMAPE
    sMAPEs = []
    print(f"sMAPE: {round(sMAPE(concatenated_outputs, concatenated_actuals), 3)}%")

    # Plot the Results
    plt.figure(figsize=(12, 5))
    plt.plot(concatenated_outputs, label="Predicted")
    plt.plot(concatenated_actuals, label="Actual")
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.savefig(f"./plots/predicted_vs_actual_12.png")
    plt.show()
