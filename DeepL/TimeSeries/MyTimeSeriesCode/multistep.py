import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torch import nn

# Load the CSV file
data_path = 'fw80_pow90_loop1.csv'
feature = 'vol_170101010.tempf'
data = pd.read_csv(data_path)
tavg = data[feature].values.reshape(-1, 1)

# Normalize the 'TAVG' variable
scaler = MinMaxScaler(feature_range=(-1, 1))
tavg_normalized = scaler.fit_transform(tavg)

# Create sequences for input-output pairs
def create_sequences(input_data, input_seq_length, output_seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - input_seq_length - output_seq_length + 1):
        x = input_data[i:(i + input_seq_length)]
        y = input_data[(i + input_seq_length):(i + input_seq_length + output_seq_length)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

input_seq_length = 10
output_seq_length = 5
X, y = create_sequences(tavg_normalized, input_seq_length, output_seq_length)

# Split the data into training and testing sets
train_size = int(len(y) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create PyTorch data loaders
train_data = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_data = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(train_data, shuffle=False, batch_size=64)
test_loader = DataLoader(test_data, shuffle=False, batch_size=64)

# Define the GRU models
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # We only care about the last output for the sequence
        return out

model = GRUModel(input_size=1, hidden_size=64, num_layers=2, output_size=output_seq_length)

# # Train the models
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 100
# for epoch in range(num_epochs):
#     for inputs, targets in train_loader:
#         optimizer.zero_grad()
#         output = model(inputs)
#         targets = targets.squeeze(dim=2) # Reshape targets to match output shape
#         loss = criterion(output, targets)
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
#
# Save the models with a specific naming convention
model_name = f"GRU_{feature}_past{input_seq_length}_future{output_seq_length}.pth"
save_path = f"models/{model_name}"
# torch.save(model.state_dict(), save_path)

# Evaluate the models
model.load_state_dict(torch.load(save_path))
model.eval()

predictions, actuals = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        output = model(inputs)
        predictions.append(output.numpy())
        actuals.append(targets.numpy())

# Reshape predictions and actuals
predictions = np.concatenate(predictions).reshape(-1, output_seq_length)
actuals = np.concatenate(actuals).reshape(-1, output_seq_length)

# Denormalize predictions and actuals
predictions_denorm = scaler.inverse_transform(predictions)
actuals_denorm = scaler.inverse_transform(actuals)

# Step 8: Visualize Results
plt.plot(actuals_denorm, label='Actual')
plt.plot(predictions_denorm, label=feature)
plt.legend()
# 将图像保存到result文件夹，并且名字为data_path_feature.png
result_path = f"results/GRU_{feature}_past{input_seq_length}_future{output_seq_length}.png"
plt.savefig(result_path)
plt.show()

# 保存预测结果到txt文件
np.savetxt('1.txt', predictions)
