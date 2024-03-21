import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256 * 3 * 3, 2304)
        self.fc2 = nn.Linear(2304, 128)
        self.fc3 = nn.Linear(128, 18)  # 18个工况分类

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = CNNModel()
