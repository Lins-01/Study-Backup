import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        # 特征提取层
        self.features = features  # 在下面的make_features函数中生成，且实例化类的时候会传入参数

        # 分类层
        # 分类层之前的展平处理一般在forward函数中进行
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            # True表示inplace，会修改输入的数据，默认为False不修改输入的数据
            # inplace=True是pytorch中的一个参数，表示是否进行覆盖运算，即是否对原数据进行修改
            # 可以降低内存的使用，加快运算速度
            # 因为会直接对输入的数据进行修改，而不是将修改后的数据另外存储到一个新的内存中
            # 同时还可以省去反复申请和释放内存的时间，加快运算速度
            # 但是会对原数据进行修改，所以会影响反向传播，因为反向传播的时候需要用到原始输入的数据
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        # 在分类层之前进行展平处理
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # 当前层是卷积层
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 当前层是线性层
            # xavier_uniform_是均匀分布初始化
            # normal_是正态分布初始化，
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 生成特征提取层
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 列表解包，*表示解包，转换为非关键字参数
    # Sequential正常用就是传入一个个非关键字参数，这里是传入一个列表，所以需要解包
    return nn.Sequential(*layers)


# 不同模型的配置
# 列表的数字代表卷积层的输出通道数，M代表最大池化层结构
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 对应图中A的网络结构
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 对应图中B的网络结构
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # 对应图中D的网络结构
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # 对应图中E的网络结构
}

# 生成vgg模型
# 第二个参数**表示解包，转换为关键字参数
# **kwargs表示关键字参数，可以传入多个关键字参数，这里是传入一个字典，所以需要解包
# 即vgg类定义的时候init函数后面要传入的参数，这里是num_classes=1000, init_weights=False
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model

# 生成vgg16模型，测试看下return nn.Sequential(*layers)，这里的layer解包后是什么样子
# 这条语句 + 在那里打个断点，debug就好了
# vgg_model = vgg(model_name='vgg13')
