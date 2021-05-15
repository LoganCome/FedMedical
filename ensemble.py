import torch
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from collections import Counter
import matplotlib.pyplot as plt
from load_data import MyDataSet
from torchvision import transforms
from torch.utils.data import Subset, sampler
import torchvision.models as models
import torch.utils.data.dataloader as dataloader

# 数据增强
data_transforms = {
    "train":
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
    "val":
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    "test":
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

}

# 读取数据集,划分数据集
img_path = "data/img/"
csv_path = "data/my_label.csv"
batch_size = 5

# 读取数据集(1094张图片)
train_set = MyDataSet(img_path, csv_path, data_transforms["train"], "train")
train_set_B = Subset(train_set, range(0, 1080))
train_loader_B = dataloader.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

# 从训练集里拿出一部分做测试
test_set = train_set
agent_test_data = 200
test_sampler = sampler.RandomSampler(data_source=train_set, num_samples=agent_test_data, replacement=True)
test_loader = dataloader.DataLoader(dataset=test_set, sampler=test_sampler, batch_size=batch_size)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on " + str(device))
print("---------------------------")

weight_1 = 0.2
weight_2 = 0.4
weight_3 = 0.2
weight_4 = 0.2


# 训练和测试
def train_and_test_1(train_loader, test_loader):
    # resnet 18
    class r18(nn.Module):
        def __init__(self):
            super(r18, self).__init__()
            self.resnet18 = models.resnet18(pretrained=True)
            self.fc1 = nn.Linear(1000, 500)
            self.fc2 = nn.Linear(500, 9)

        def forward(self, x):
            x = self.resnet18(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # resnext 50
    class rx50(nn.Module):
        def __init__(self):
            super(rx50, self).__init__()
            self.resnext50 = models.resnext50_32x4d(pretrained=True)
            self.fc1 = nn.Linear(1000, 500)
            self.fc2 = nn.Linear(500, 9)

        def forward(self, x):
            x = self.resnext50(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # resnet 34
    class r34(nn.Module):
        def __init__(self):
            super(r34, self).__init__()
            self.resnet34 = models.resnet34(pretrained=True)
            self.fc1 = nn.Linear(1000, 500)
            self.fc2 = nn.Linear(500, 9)

        def forward(self, x):
            x = self.resnet34(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    # mobilenet_v2
    class mv2(nn.Module):
        def __init__(self):
            super(mv2, self).__init__()
            self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            self.fc1 = nn.Linear(1000, 500)
            self.fc2 = nn.Linear(500, 9)

        def forward(self, x):
            x = self.mobilenet_v2(x)
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    epochs = 100
    # 不同的模型使用不同的lr
    r18_lr = 0.0001
    rx50_lr = 0.00001
    r34_lr = 0.0001
    mv2_lr = 0.0001

    # losslist
    r18_losslist = []
    rx50_losslist = []
    r34_losslist = []
    mv2_losslist = []
    E_losslist = []

    # 四个模型
    model1 = r18()
    model2 = rx50()
    model3 = r34()
    model4 = mv2()

    # 四个模型都在GPU运行（显存不够可以调小batch_size）
    model1.to(device)
    model2.to(device)
    model3.to(device)
    model4.to(device)

    # 四个模型使用同一个损失函数
    loss_func = nn.CrossEntropyLoss()  # 损失函数的类型：交叉熵损失函数

    # 四个模型使用不同的优化器
    optimizer1 = optim.Adam(model1.parameters(), lr=r18_lr)  # Adam优化，也可以用SGD随机梯度下降法
    optimizer2 = optim.Adam(model1.parameters(), lr=rx50_lr)
    optimizer3 = optim.Adam(model1.parameters(), lr=r34_lr)
    optimizer4 = optim.Adam(model1.parameters(), lr=mv2_lr)
    # optimizer1 = optim.RMSprop(model1.parameters(), lr=lr)    # 也许可以考虑不同的模型使用不同的优化算法

    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=(epochs // 9) + 1)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=(epochs // 9) + 1)
    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=(epochs // 9) + 1)
    scheduler4 = optim.lr_scheduler.CosineAnnealingLR(optimizer4, T_max=(epochs // 9) + 1)

    for epoch in range(epochs):
        flag = 0
        for images, labels in train_loader:
            images = images.to(device)
            # print(images.size())    # [100, 3, 224, 224]
            labels = labels.to(device)

            # 四个模型各有一个output
            output1 = model1(images)
            output2 = model2(images)
            output3 = model3(images)
            output4 = model4(images)
            # 模型聚合
            outputE = weight_1 * output1 + weight_2 * output2 \
                      + weight_3 * output3 + weight_4 * output4
            # print(output1.size())   # [100, 5]
            # print(labels.size())   # [100]

            # 四个模型各有一个loss
            loss1 = loss_func(output1, labels)
            loss2 = loss_func(output2, labels)
            loss3 = loss_func(output3, labels)
            loss4 = loss_func(output4, labels)
            lossE = loss_func(outputE, labels)

            optimizer1.zero_grad()  # 清空梯度
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            loss1.backward()  # 误差反向传播，计算参数更新值
            loss2.backward()
            loss3.backward()
            loss4.backward()

            optimizer1.step()  # 将参数更新值施加到net的parameters上
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()

            scheduler1.step()  # 更新lr
            scheduler2.step()
            scheduler3.step()
            scheduler4.step()

            # 查看每轮损失函数具体的变化情况
            if (flag + 1) % 30 == 0:
                print('Epoch [{}/{}], Loss1: {:.4f}, Loss1: {:.4f}, Loss1: {:.4f}, Loss1: {:.4f}'
                      .format(epoch + 1, epochs, loss1.item(), loss2.item(), loss3.item(), loss4.item()))
                # 五个losslist的更新
                r18_losslist.append(loss1.item())
                rx50_losslist.append(loss2.item())
                r34_losslist.append(loss3.item())
                mv2_losslist.append(loss4.item())
                E_losslist.append(lossE.item())

            flag += 1

    # 测试，评估准确率
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # print(images.size())
        # print(labels.size())

        # 四个模型各有一个output
        output1 = model1(images)
        output2 = model2(images)
        output3 = model3(images)
        output4 = model4(images)
        # 模型聚合
        outputE = weight_1 * output1 + weight_2 * output2 \
            + weight_3 * output3 + weight_4 * output4

        values, predicte = torch.max(outputE, 1)  # 0是每列的最大值，1是每行的最大值
        total += labels.size(0)
        # predicte == labels 返回每张图片的布尔类型
        correct += (predicte == labels).sum().item()

        # print(100 * correct / total)

    print("The accuracy of total {} images: {}%".format(total, 100 * correct / total))

    return r18_losslist, rx50_losslist, r34_losslist, mv2_losslist, E_losslist


# 训练测试
lossList1, lossList2, lossList3, lossList4, lossListE = train_and_test_1(train_loader_B, test_loader)

# 保存模型


# 绘制本地误差曲线
plt.plot(lossList1, label='ResNet18')
plt.plot(lossList2, label='ResNeXt50')
plt.plot(lossList3, label='COVID-Net')
plt.plot(lossList4, label='MobileNetV2')
plt.plot(lossListE, label='ensemble')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()
