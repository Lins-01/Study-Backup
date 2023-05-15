import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34


# 使用官方提供的resnet模型，ctrl+左键，下面的resnet，可以看到源码
# import torchvision.models.resnet
# 源码里面有下载预训练模型权重的链接，放到网页就能下载了。
# 一般最上面，或者搜url就能看到类似下面这样的东西 ，里面的链接就是我们要的
# resnet这个有很多个不同的版本，我们这里选34的下载就好了
# url="https://download.pytorch.org/models/resnet34-b627a593.pth"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     # 标准化处理的时候这里的标准化处理的参数都是来自官网提供的一个tranform learning的一个教程
                                     # 那个教程也用的resnet网络的权重，他这里也就照搬过来了
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),  # 这里是将原图片长宽比不动，将他最小边长缩放到256，也就学官网的教程的，
                                   # 也直接ctrl看resize这个函数的说明
                                   # 再使用中心裁剪，将图片裁剪成224*224的大小
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # 上面的方法，是获取到上层目录的路径，然后拼接/data_set/flower_data，这样就能得到数据集的路径了
    # 那我直接给他路径
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

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)  # linux系统将线程个数设置为>0的个数，就能加速图像预处理的过程，也就是图像预处理的时候，可以同时处理多张图片

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnet34()

    # load pretrain weights 导入预训练模型权重的方法（官方给的）
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # 载入预训练模型的权重
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure

    # net.fc也就是model.py中定义网络init中，self.fc定义的fc
    # in_features是输入特征矩阵的深度，也就是fc层的输入
    in_channel = net.fc.in_features
    # 由于花分类数据集是5分类，所以这里将fc层的输出改成5
    # nn.Linear重新赋值全连接层，in_channel是输入，5是输出
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train() # 和net.eval()是控制batch normalization的，不要忘了
        # batch normalization在训练和测试的时候是不一样的
        # 训练的时候是用的batch的均值和方差，测试的时候是用的整个数据集的均值和方差
        # 所以执行方法也不一样，训练的时候是net.train()，测试的时候是net.eval()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
