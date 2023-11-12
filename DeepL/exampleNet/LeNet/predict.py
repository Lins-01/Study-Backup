import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    # transforms.Resize()调整图像大小
    # 因为我们网上下载的用来预测图片大小不标准
    # 而我们的网络（LeNet）定义的是固定输入32*32的图片，所以需要调整一下
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 训练过程中需要标准化，预测也需要

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    # 加载模型
    # 加载权重文件
    net.load_state_dict(torch.load('Lenet.pth'))

    # 读取图片 ，用的PIL库
    # 一般用PIL或numpy读取图片，他一般的图像格式shape是HWC， 即高度、宽度、通道数，（就是矩阵存储的顺序不一样啦）
    # 而pytorch的图像格式是CHW，需要转换一下
    im = Image.open('1.jpg')
    # 预处理
    im = transform(im)  # [C, H, W]
    # 这里加上一个维度batch，因为pytorch的输入[batch, channel, height, width]
    # torch.unsqueeze()函数的作用是将一维变二维，二维变三维，以此类推
    # dim=0表示在第0维前增加一个维度
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    # 预测
    # 同样不需要求损失梯度
    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
        # 这里用softmax也可以
        # predict = torch.softmax(outputs, dim=1)
        # 不过这里的predict是个tensor，需要再取出最大值，再放入classes中
    # 将最大的那个值的索引取出来，即为预测的类别
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
