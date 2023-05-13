import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def main():
    # 对数据/图像进行预处理的函数
    # 此transform非彼transformer，这里就是转换的意思，单独的一个包
    # transforms.Compose()函数的功能是将多个transform组合起来使用
    # 这里将两个预处理方法打包起来使用，这样在后面的代码中就可以直接调用这个函数了
    # transforms.ToTensor()将PILImage或者numpy.ndarray转化为tensor
    # transforms.Normalize()对数据按通道进行标准化，即先减均值，再除以标准差
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 下载数据集
    # 50000张训练图片,10个类别，每类5000张图片
    # download=True表示如果本地没有下载数据，就下载数据，下载完就可以设置为False了
    # torchvision.datasets下面还有很多别的数据集，看大写开头的就是
    train_set = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,  # 下载训练集
                                             download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=36,  # 每次训练的图片数
                                               shuffle=True,  # 是否打乱训练集
                                               num_workers=0)  # 使用线程数，在windows下只能设置为0，linux下可自定义

    # 10000张测试图片
    val_set = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,  # 下载测试集，但下载完训练集后，测试集就已经下载好了
                                           download=False, transform=transform)
    # 加载数据集
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=10000,  # 一次性加载所有的测试数据
                                             shuffle=False,
                                             num_workers=0)
    # 获取测试集中的图像和标签，用于accuracy计算
    # 用iter()函数创建一个迭代器
    val_data_iter = iter(val_loader)
    # val_image, val_label = val_data_iter.next()
    # 版本问题报错
    # 改如下即可
    # 转换为迭代器之后，就可以通过next()方法获取一批数据
    # 数据中就包含了数据的图像，以及对应的标签值label
    val_image, val_label = next(val_data_iter)

    # 接着将classes的标签导入
    # 元组类型，值不能改
    # 对应的标签索引[0]就是plane ， [1]就是car.....
    # classes = ('plane','car','bird','cat',
    #            'deer','dog','frog','horse','ship','truck')

    """
    # 简单看一下数据集中的图片
    # 显示图片
    def imshow(img):
        img = img / 2 + 0.5 # unnormalize 反标准化处理，就是标准化的公式逆转，化简
        npimg = img.numpy() # 将tensor转化为numpy
        plt.imshow(np.transpose(npimg,(1,2,0))) # 将tensor的维度转化为(1,2,0)
        plt.show()

    #print labels
    print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
    # 显示图片
    imshow(torchvision.utils.make_grid(test_image))
    """
    # 加载模型 即实例化这个类
    net = LeNet()
    # 定义损失函数
    # 损失函数是softmax+log+NLLLoss()，也就是负对数似然损失函数
    loss_function = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(net.parameters(),  # 优化器优化的参数，也就是我们需要训练的参数
                           lr=0.001  # 学习率
                           )
    # 训练过程
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0  # 累加训练过程中的损失

        # 遍历训练集样本

        # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
        # 同时列出数据和数据下标，一般用在for循环当中
        # start=0表示下标从0开始
        for step, data in enumerate(train_loader, start=0):
            # 得到数据后将其分离，一部分是输入，一部分是标签
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # 将历史损失梯度置为0，否则会累加
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # 将得到的输入图片，传入网络，进行正向传播，得到输出
            outputs = net(inputs)
            # 计算损失
            # 第一参数是网络的输出即预测值，第二个参数是图片真实的标签
            loss = loss_function(outputs, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 处理一些打印
            # print statistics

            # loss累加
            # item()函数是将一个标量Tensor转换成一个python number
            running_loss += loss.item()

            # 每隔500步打印一次信息
            if step % 500 == 499:  # print every 500 mini-batches
                # with是一个上下文管理器，用于简化try...finally...的写法

                # 在这里是为了不让梯度在验证集上反向传播，因为验证集不需要反向传播
                # 因为反向传播会 1. 占用更多算力，消耗更多资源
                # 2. 要存储每个节点的损失梯度，占用更多内存
                # 在验证集上只需要前向传播，计算损失，计算准确率即可
                # with torch.no_grad():表示在这个上下文环境中，不计算梯度
                with torch.no_grad():
                    # 前向传播

                    outputs = net(val_image)  # [batch, 10]
                    # 这里outputs是什么
                    # outputs是网络的输出，是一个[batch,10]的tensor
                    # 10是什么意思
                    # 10是类别的个数，因为我们的数据集是CIFAR10，一共有10个类别

                    # 寻找输出的最大的index在什么位置，即网络预测最可能的类别 ，torch.max来实现

                    # torch.max()函数有两个返回值，第一个是最大值，第二个是最大值的索引
                    # 第一个参数是一个tensor，第二个参数是dim，表示在哪个维度上寻找最大值
                    # 这里在第一个维度上寻找最大值，即在每个样本上寻找最大值
                    # 第0个维度是batch，第1个维度是类别
                    # 这里就是[batch,10]里面的10个类别里面找最大值，即找到最有可能的类别，即找到最大值的索引

                    # [1]表示只取最大值的索引，不取最大值，因为不需要，我只需要是哪个类别就好了
                    predict_y = torch.max(outputs, dim=1)[1]

                    # torch.eq 将真实的标签和预测的标签进行比较，相同的话返回1，不同的话返回0
                    # 再通过sum()函数将所有的1加起来，即得到预测正确的个数
                    # 因为整个过程都是在tensor上进行的，所以最后要通过.item()函数将tensor转换成python number
                    # 再除以总的测试样本数，即得到准确率
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    # 打印信息
                    # 训练到多少轮、多少步、损失、准确率
                    # 500步训练的平均误差/损失，即running_loss/500
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # 将running_loss置为0，为下一个500步做准备
                    running_loss = 0.0

    # 全部训练完成后，打印信息
    print('Finished Training')

    # 保存模型
    save_path = './Lenet.pth'
    # 保存模型的参数
    # net.state_dict()是一个字典，里面保存了网络中每个层的参数
    # torch.save()函数可以将其保存到硬盘上
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
