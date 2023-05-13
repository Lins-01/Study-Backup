# 使用torch.nn包来构建神经网络.
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):  # 继承于nn.Module这个父类
    def __init__(self):  # 初始化网络结构
        super(LeNet, self).__init__()  # 多继承需用到super函数

        """
        参数分别表示：
            卷积核的深度channel（个数）=输入特征矩阵的深度
            卷积核的个数=输出特征矩阵的深度（个数），
            卷积核的大小
            
            步长和padding是默认值
            padding默认是0，即不补0，为1即补一圈0；
                还可以是tuple型如(2, 1) 代表在上下补2行，左右补1列。
            步长默认是1
        """
        self.conv1 = nn.Conv2d(3, 16, 5)
        """
        # 参数是池化核的大小和步长
        # 步长默认是kernel_size
        """
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        """
        # 全连接层
        # 参数是输入和输出的维度
        # Linear(in_features, out_features, bias=True)
       
        # 这些维度是多少，都是你看模型的结构图里面，他写的是多少，你就写多少
        """
        # 看模型结构知道，第一个全连接层的输出是120
        # 输入是上一层的输出拉成一位，维度就是这么多
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # 看模型结构知道，第二个全连接层的输出是84
        self.fc2 = nn.Linear(120, 84)
        # 看模型结构知道，第三个全连接层的输出是10
        # 最后一层这里的输出其实要根据训练集来修改的
        # 这次任务用的cifar10数据集，是具有10个类别的，所以最后一层输出是10
        self.fc3 = nn.Linear(84, 10)

        # x就是输入的数据，就是
        # pytorch 中 tensor（也就是输入输出层）的 通道排序为：[batch, channel, height, width]

    def forward(self, x):  # 正向传播过程
        # 通过这一步卷积过程，得到output，得到16个特征矩阵
        # 这里的注释中input没写batch_size
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        # 通过这一步池化过程，得到output，深度不变，把尺寸（高度和宽度）缩放了一半
        x = self.pool1(x)  # output(16, 14, 14)

        # 一直都是对x操作，x上一步之后的深度是16，所以这里的输入深度是16
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        #  再缩减为一半
        x = self.pool2(x)  # output(32, 5, 5)

        # 全连接层的输入是一个一维向量，所以要把上一层的输出拉成一维向量
        # 第一个参数-1代表自动计算这个维度的大小，
        # 第一个维度是batch_size，所以这里的-1就是batch_size的大小
        # view的第一个参数是batch_size，第二个参数是其他维度的乘积
        # 还是不太懂这里-1为什么推理出是batch_size的大小？？？
        # 32*5*5是展平后的维度
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)

        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        # 这里为什么没有用softmax函数？？？
        # 一般分类问题最后一层都是用softmax函数的
        # 但这里因为在交叉熵函数中已经包含了一个高效的softmax函数
        return x




# # 测试一下model
# import torch
#
# input1 = torch.rand([32, 3, 32, 32])
# # 实例化一个model
# model = LeNet()
# # 打印一下model的结构
# print(model)
# output = model(input1)

