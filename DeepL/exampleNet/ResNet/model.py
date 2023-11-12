import torch.nn as nn
import torch


# ResNet18/34的残差结构，用的是2个3x3的卷积


class BasicBlock(nn.Module):
    expansion = 1  # 残差结构中，主分支的卷积核个数是否发生变化，不变则为1

    # 18/34层中的残差结构，conv2中两层里面的各自的卷积核之间个数一样，都是64，即不变

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):  # downsample是下采样,对应虚线的残差结构
        # python传参可以传函数，这里的downsample就是一个函数，用于将输入x的通道数变为out_channel
        # 在每个conv2、conv3中，第一个残差结构都是虚线残差架构，因此需要下采样

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  # 如果是虚线残差结构，需要下采样
            identity = self.downsample(x)  # 下采样的作用是将x的通道数变为out_channel

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # identity是shortcut的输出，即x，out是主分支
        out = self.relu(out)

        return out


# ResNet50/101/152的残差结构，用的是1x1、3x3、1x1的卷积
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 残差结构中第三层卷积核个数是第一/二层卷积核个数的4倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # 这里其实用的resnext的残差结构（他后面在github加的）
        # 这里是用了组卷积，groups是分组数，width_per_group是每组的卷积核个数
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # 残差结构 会根据定义的不同层结构传入不同的残差结构，这里是BasicBlock（18/34层）或者Bottleneck（50/101/152层）
                 blocks_num,  # block_num为残差结构中conv2_x~conv5_x中残差块个数，是一个列表（34层时是[3, 4, 6, 3]）
                 num_classes=1000,  # 分类数
                 include_top=True,  # 方便以后在resnet网络基础上搭建更加复杂的结构，比如faster rcnn ？？
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 这里的输入通道数是64，输入指的是残差结构的输入，也就是maxpooling之后的输出=conv1的输出
        # 是通道数，不是output_size（output_size是输出的特征图的尺寸，conv1需要是112x112）

        self.groups = groups
        self.width_per_group = width_per_group

        # 7x7的卷积层
        self.conv1 = nn.Conv2d(3,  # 输入通道数 RGB图像
                               self.in_channel, kernel_size=7, stride=2,
                               padding=3,  # 保证输入输出尺寸相同，和stride一起达到下采样的效果，缩减尺寸
                               # 7x7的卷积核，padding=3，输出尺寸为（n+2padding-filter）/stride+1 = (224+6-7)/2+1 = 112
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=1)  # 3x3的池化层，输出尺寸为（n+2p-f）/s+1=（112+2*1-3）/2+1 = 56
        # layer1是conv2_x对应的一系列残差结构
        # 通过_make_layer函数生成layer1，layer2，layer3，layer4
        self.layer1 = self._make_layer(block, 64, blocks_num[0])  # blocks_num[0]是conv2_x中残差块的个数
        # layer2是conv3_x对应的一系列残差结构
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # layer3是conv4_x对应的一系列残差结构
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # layer4是conv5_x对应的一系列残差结构
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # include_top为True时，表示在resnet基础上加上全连接层，用于分类
        if self.include_top:
            # 平均池化下采样层，这里通过一个自适应的平均池化层
            # 无论你输入的图像尺寸是多少，都会输出宽高是（1，1）的特征图
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接层，输出节点层
            # 上面特征矩阵都展平为（1，1）了，那下面的全连接层的输入节点数就是特征矩阵的深度
            # 当18/34时就是512，50/101/152时就是2048=512*4
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        # 对卷积层进行一个初始化操作，kaiming_normal_是一个初始化方法，初始化权重
        # 因为上面是刚定义好了网络结构，还没有给权重赋值，所以这里需要初始化
        # 初始化权重还没看到细讲，看下前面的
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block,
                    channel,  # 残差结构中第一层卷积层使用的卷积核个数 ， 后面层是倍数关系表示
                    block_num,  # 该层残差结构的个数
                    stride=1):
        downsample = None
        # layer1调用时，也就是conv2时，18/34的不执行，50/101/152的执行，但也只是改变深度，不改变尺寸，因为stride=1
        # layer2、3、4调用时，也就是conv3/4/5时，18/34/50/101/152全部执行，因为stride全部不为1，第一个都是虚线残差结构
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 定义一个下采样函数，用于处理输入输出通道数不一致的情况，在下面的layer.append里面传入block
            # nn.Sequential()是一个有序容器，传入的是一个有序的模块，会按照传入的顺序依次执行
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,  # 输入通道数,输入特征矩阵的深度
                          # layer1调用时，都是64，执行到后面self.in_channel = channel * block.expansion，会变为64*1=64或者64*4=256
                          # 所以当下一层调用时，就已经是该层需要的输入通道数了

                          channel * block.expansion,  # 每个残差结构的输出通道数 ，也是捷径分支的输出通道数
                          kernel_size=1,  # 50/101/152层第一个虚线残差都是用的1x1卷积核，宽高不变
                          stride=stride,  # 当为conv2时，它的虚线残差结构结构，只用调整深度，不用调整尺寸
                          # 当为conv3/4/5时，它的虚线残差结构结构，既要调整深度，也要调整尺寸
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 把第一个残差结构放入layers
        layers.append(block(self.in_channel,
                            channel,  # 残差结构中第一层卷积层使用的卷积核个数
                            downsample=downsample,  # 上面定义的下采样函数，
                            # 当18/34层时，downsample=None，所对应的就是个实线残差结构
                            # 50/101/152层时，downsample不为None，所对应的就是个虚线残差结构，深度变4倍，宽高可能变
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 18/34层时，self.in_channel = channel * 1 = 64/128/256/512
        # 50/101/152层时，self.in_channel = channel * 4 = 256/512/1024/2048

        # 通过for循环，将剩下的实线残差结构添加入layers
        for _ in range(1, block_num):  # 从1开始，因为第一个已经添加进去了
            layers.append(block(self.in_channel,  # 上面一步处理过了，并不是一直等于64
                                channel,  # 残差结构中第一层卷积层使用的卷积核个数，后面的用倍数关系表示。bottoleneck中会处理
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # *layers表示将layers解包，layers中的元素依次作为参数传入nn.Sequential()中
        # *号，将list列表转化为非关键字参数形式传进去，按照位置传递给函数形参
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
