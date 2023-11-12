import os
import sys
import json


import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

# 记录一次epoch时间
import time

def main():

    # 有GPU就去使用第一块GPU，没有就使用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 数据预处理
    # 不过写成了字典形式，这样就可以在训练和验证的时候使用不同的数据预处理方式
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 随机水平翻转——》数据增强？
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = "E:/Document/CodeSpace/Data_set/flower_data"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw) # 0表示不使用额外的进程来加速读取数据，用主线程来读取数据

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))


    # 用来显示图片，简单看下数据集
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # 调试用的，查看模型的参数
    # pata = list(net.parameters())

    # 优化器
    # 优化对象：net.parameters()，网络中所有的可学习参数
    # 学习率：0.0002，也是up测试得到的效果不错的学习率，调大调小准确率都会下降
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'

    # 定义一个最好的准确率，用来保存后面训练中准确率最高的模型
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train

        # dropout和batch normalization都是在训练的时候使用，测试的时候不使用
        # dropout也是只要在训练时候的正向传播时候丢弃神经元，测试的时候就不丢弃了
        # 训练的时候使用net.train()打开dropout，测试的时候使用net.eval()关闭dropout
        net.train()

        # 统计训练过程中的平均损失值
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        # 记录训练开始时间
        t1 = time.perf_counter()

        # 遍历数据集数据集，将数据分为输入和标签
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()   # 梯度清零，否则会累加

            # images.to(device)将数据放到GPU上
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # 显示本次epoch所需时间
        print(time.perf_counter() - t1)
        # validate 训练完一轮之后进行验证
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # 这里不需要计算梯度，所以使用torch.no_grad()
        # torch.no_grad()禁止pytorch对参数的跟踪
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 保存准确率最高的模型
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
