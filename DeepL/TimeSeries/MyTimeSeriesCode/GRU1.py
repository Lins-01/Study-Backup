import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch import nn

# Step 1: Load and Normalize Data
data_path = 'fw80_pow90_loop1.csv'
data = pd.read_csv(data_path)
feature = 'vol_170101010.tempf'
# values作用将值都转为numpy数组 ndarray
# reshape(-1, 1)作用将数组转为2D的，因为MinMaxScaler是用来不同特征之间的归一化的，所以默认多个特征，即多列，所以需要转为2D
# 归一化是因为数据的范围不一样，归一化后，数据的范围都在0-1之间，消除特征之间的量纲差异/消除奇异样本（比其他大/小很多的）数据导致的不良影响，提高模型的性能和稳定性
tavg = data[feature].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
tavg_normalized = scaler.fit_transform(tavg)

# Step 2: Create Sequences
def create_sequences(data, seq_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]

        # 只预测一步 y = data[i+seq_length]
        # 预测多步
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    # 若data长度为8175（即样本数8175），seq_length为10，那么xs的长度为8165，ys的长度为8165
    # date维度为(8175, 1)，xs维度为(8165, 10, 1)，ys维度为(8165, 1)  y = data[i+seq_length]
    # 如果只预测一步，那么ys的维度为(8165, 1)  y = data[i+seq_length]
    # 如果改为预测多步，那么ys的维度为(8165, 10, 1)  y = data[i+seq_length:i+seq_length+output_seq_length]，10表示预测10步，1表示一个特征
    # 1表示一个特征
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(tavg_normalized, seq_length)

# Step 3: Split Data
train_size = int(len(y) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 4: Create Data Loaders
# TensorDataset作用是将数据转为tensor格式，因为pytorch的数据格式是tensor
# 训练的输入和label一起放入作为train_data
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
# 测试的输入和label一起放入作为test_data
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
# DataLoader作用生成迭代器，每次迭代都会返回一个batch_size大小的数据
train_loader = DataLoader(train_data, shuffle=False, batch_size=64)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)
print("train_data: ", len(train_data))
print("test_data: ", len(test_data))

# 到这里都还没把数据放到GPU上，默认的还是在cpu上
# Step 5: Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size): # 这里的参数是类实例化的时候传入
        super(GRUModel, self).__init__()
        # batch_first=True表示then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). 默认是False
        # 模型的输入在forward(self,x)的x中传入
        # 模型的输入描述如上，除此之外，一个batch中不同样本的序列长度可以不同，具体见下面官方描述原文
        # The input can also be a packed variable length sequence.
        # See torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence() for details.
        # 那GRU是怎么处理每个序列的呢？
        # 训练时，每个序列有一些时间步，每个时间步都有一个输入

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x): # 这里的参数是模型调用的时候传入
        out, _ = self.gru(x) # gru和lstm都有两个输出，一个是out，一个是hidden，这里只用到了out，out的维度为(batch_size, seq_length, hidden_size)
        # 第一个是
        out = self.fc(out[:, -1, :])
        return out

model = GRUModel(input_size=1, hidden_size=64, num_layers=2, output_size=1)

# Step 6: Train the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Save the models
torch.save(model.state_dict(), 'model.pth')

# Step 7: Evaluate the Model
model.load_state_dict(torch.load('model.pth'))
model.eval()

predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        output = model(inputs)
        predictions.append(output.numpy())
        actuals.append(targets.numpy())

predictions = np.concatenate(predictions).reshape(-1, 1)
actuals = np.concatenate(actuals).reshape(-1, 1)

# Denormalize predictions
# 直接用前面minmaxscaler实例化的scaler的inverse_transform函数，进行反归一化
predictions_denorm = scaler.inverse_transform(predictions)
actuals_denorm = scaler.inverse_transform(actuals)

# Step 8: Visualize Results
plt.plot(actuals_denorm, label='Actual')
plt.plot(predictions_denorm, label=feature)
plt.legend()
# 将图像保存到result文件夹，并且名字为data_path_feature.png
save_name=feature+".png"
plt.savefig("1.png")
plt.show()
