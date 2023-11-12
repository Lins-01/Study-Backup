import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # nn.Sequential() 顺序容器，模块将按照构造函数中传递的顺序添加到模块中。
        # 将一系列层结构打包，组合成一个新结构。为什么要这样呢？
        # 因为在深度学习中，我们经常需要写很多层，而且这些层之间还有顺序关系，这样写起来很麻烦。
        # 之前是写很多个self.xx = nn.xx()，现在可以直接写在Sequential里面，这样就简单多了。

        # 这一组卷积层和池化层，用来提取图像特征
        self.features = nn.Sequential(

            # 这里参数都是减少为原论文一半，因为他测试之后发现效果一样
            # 但是这样我们计算量减少了一半，可以加快训练

            # padding传入tuple(1,2)时，表示在上下方各补一行0，在左右方各补两列0
            # 实现更精细化padding操作nn.ZeroPad2d
            # nn.ZeroPad2d((1, 2, 1, 2))表示在左侧补一列0，在右侧补两列0，在上方补一行0，在下方补两行0
            # pytorch中按照公式计算不为整数时，向下取整，会将余数舍弃，所以这里padding=2也可以实现
            # 具体可以看他之前写的一个博客：https://blog.csdn.net/qq_37541097/article/details/102926037
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]

            # inplace=True是pytorch中的一个参数，表示是否进行覆盖运算，即是否对原数据进行修改
            # 可以降低内存的使用，加快运算速度
            # 因为会直接对输入的数据进行修改，而不是将修改后的数据另外存储到一个新的内存中
            # 同时还可以省去反复申请和释放内存的时间，加快运算速度
            # 但是会对原数据进行修改，所以会影响反向传播，因为反向传播的时候需要用到原始输入的数据
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )

        # 这一组全连接层，用来分类
        # 分类器
        self.classifier = nn.Sequential(
            # 这里是在MaxPool2d展平之后，展平操作在forward函数中
            # dropout一般用在全连接层与全接连层之间，让他部分神经元失活，防止过拟合
            # p默认为0.5，表示失活的比例
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            # dropout一般用在全连接层与全接连层之间，让他部分神经元失活，防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            # 输出是类别的个数num_classes
            nn.Linear(2048, num_classes),
        )

        # 初始化权重
        # 搭建网络时候传入init_weights=True，就会调用这个函数

        # 其实并不需要初始化方法，因为pytorch中的nn.Conv2d()和nn.Linear()都有默认的初始化方法
        # pytorch中的nn.Conv2d()默认的初始化方法是kaiming_normal_()，何凯明初始化方法
        # nn.Linear()默认的初始化方法是xavier_normal_()，xavier初始化方法
        if init_weights:
            self._initialize_weights()

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.features(x)
        # 展平处理，即将多维的输入一维化，常用在卷积层到全连接层的过渡
        # torch.flatten()函数的作用是将输入的tensor展平成一维
        # 展平的维度是start_dim=1，表示从第一维开始展平，第0维是batch_size，不需要展平
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 首先遍历self.modules()，然后判断m是否是nn.Conv2d，如果是就初始化权重
        # self.modules()是继承自nn.Module的方法，会返回一个迭代器，包含网络中的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 如果该层是卷积层，就用kaiming_normal_方法初始化权重
                # 权重的初始化方法有很多种，这里使用的是kaiming_normal_方法,何凯明初始化方法
                # m.weight表示权重，mode='fan_out'表示权重的形状，nonlinearity='relu'表示使用relu激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 如果该层是全连接层，就用normal_方法初始化权重
                # 也就是正态分布初始化方法，均值为0，标准差为0.01
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
