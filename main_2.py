import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import datetime
import data_split2
from extract_dataset import extract_dataset
from train_test import label_smoothing
from train_test import distillation_loss
from train_test import train_and_valid

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 让程序报错

# 定义数据集路径即模型保存路径
# data_dir = '/home/liuz/xingy_project/home/liuz/xingy_project/data/'  # 数据路径
# models_dir = '/home/liuz/xingy_project/home/liuz/xingy_project/models2/'  # 模型保存路径
# save_dir = '/home/liuz/xingy_project/home/liuz/xingy_project/output2/'  # 结果保存路径

# 获取代码文件所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 定义相对路径
data_dir = os.path.join(base_dir, 'data')  # 数据路径
models_dir = os.path.join(base_dir, 'models2')  # 模型保存路径
save_dir = os.path.join(base_dir, 'output2')  # 输出保存路径

# 数据处理
(initial_train_data, initial_train_target, initial_test_data, initial_test_target,
 incremental_train_data, incremental_train_target, incremental_test_data,
 incremental_test_target) = data_split2.split_data(data_dir)

# 定义数据增强和标准化步骤
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# 参数的意义：[0.485, 0.456, 0.406] 是图像每个RGB通道的均值；[0.229, 0.224, 0.225] 是图像每个通道的标准差。
# 作用：对图像进行归一化，加速模型的训练，有利于避免过拟合
transform_train = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(),  # 随机翻转图像的上下左右镜像
    transforms.ToTensor(),  # 将图像转换为张量
    normalize])

transform_test = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    normalize])

# 参数列表
batch_size = 128  # 训练批次大小
num_classes = 0  # 当前分类类别总数
base_classes = 10
incre_classes = 10  # 每个任务的新增类别数
num_epochs_base = 20  # 迭代轮次
num_epochs_incre = 20
num_extract = 50  # 抽取旧数据量
task_num = 10  # 任务数量
istype = 2
device = torch.device('cuda:0')
# device = torch.device('cpu')

for i in range(task_num):
    if i == 0:
        print('initial task start\n')
        train_ds = torch.utils.data.TensorDataset(torch.tensor(initial_train_data),
                                                  torch.tensor(initial_train_target))
        test_ds = torch.utils.data.TensorDataset(torch.tensor(initial_test_data),
                                                 torch.tensor(initial_test_target))

        train_data = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)
        merged_train_target, merged_train_data = initial_train_target, initial_train_data

        ### 迁移学习
        # 下载resnet50的预训练模型
        resnet50 = models.resnet50(pretrained=True)
        fc_inputs = resnet50.fc.in_features  # 获取Resnet50模型最后一层输出大小
        num_classes = num_classes + incre_classes
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 1024),  # 线性层，输入参数改为隐藏层的输出大小
            nn.ReLU(),  # 激活函数，用于分离线性和非线性
            nn.Dropout(0.3),  # 随机丢弃层，p=0.4为随机删除神经元的概率
            nn.Linear(1024, num_classes, bias=False),  # 将输出特征数量更改为分类数量
            nn.LogSoftmax(dim=1)  # 归一化
        )
        resnet50 = resnet50.to(device)
        # print(resnet50.fc.out_features)
    else:
        print('incremental task{} start\n'.format(str(i)))
        train_ds = torch.utils.data.TensorDataset(torch.tensor(merged_train_data),
                                                  torch.tensor(merged_train_target))
        test_ds = torch.utils.data.TensorDataset(torch.tensor(incremental_test_data[i - 1]),
                                                 torch.tensor(incremental_test_target[i - 1]))

        train_data = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
        test_data = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False)

        # 先将新模型的fc输出层置为上一轮类别数
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes, bias=False),
            nn.LogSoftmax(dim=1)
        )
        resnet50 = resnet50.to(device)

        # 加载已经训练好的模型
        resnet50.load_state_dict(torch.load(models_dir + 'rd_' + str(i) + '_task' + str(i) + '_model.pt'))
        resnet50.eval()

        # 修改模型fc层为新类别数
        num_classes += incre_classes
        resnet50.fc = nn.Sequential(
            nn.Linear(fc_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes, bias=False),
            nn.LogSoftmax(dim=1)
        )
        resnet50 = resnet50.to(device)
        # 定义损失函数和优化器。

    loss_func = nn.NLLLoss()  # 交叉熵损失函数的最后一步，取负求和
    if i > 0:
        optimizer = optim.Adam(resnet50.parameters(), lr=0.001)  # 管理并更新模型中可学习的参数的值
    else:
        optimizer = optim.Adam(resnet50.parameters())

    train_data_size = len(train_ds)
    valid_data_size = len(test_ds)
    if i == 0:
        num_epochs = num_epochs_base
    else:
        num_epochs = num_epochs_incre
    trained_model, history = train_and_valid(resnet50, train_data, test_data, device, loss_func, optimizer, num_epochs,
                                             i, train_data_size, valid_data_size)
    # 冻结训练过的fc层
    for param in resnet50.fc.parameters():
        param.requires_grad = False

    # 保存模型
    torch.save(history, models_dir + 'rd_' + str(i + 1) + '_history.pt')
    torch.save(trained_model.state_dict(),
               models_dir + 'rd_' + str(i + 1) + '_task' + str(i + 1) + '_model.pt')
    if i < task_num - 1:
        merged_train_target, merged_train_data = extract_dataset(i, base_classes, incre_classes, num_extract, data_dir, istype,
                                                                 merged_train_target, merged_train_data)

    history = np.array(history)

    # 绘制图像
    current_timestamp = time.time()
    dt = datetime.datetime.fromtimestamp(current_timestamp)
    dt = str(dt)

    m = int(len(history) / 5)

    # 绘制loss变化曲线
    # plt.plot(history[:, 0], marker='o')
    # for j in range(m):
    #     plt.text(j * 5, history[j * 5, 0], f"({j * 5}, {history[j * 5, 0]:.2f})",
    #              horizontalalignment='left', verticalalignment='bottom')
    #
    # plt.legend(['Tr Loss'])
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Loss')
    # plt.savefig(dt + '_loss_curve_task_' + str(i + 1) + '.png')  ##
    # plt.show()
    # plt.clf()

    # 绘制准确率的变化曲线
    plt.plot(history[:, 1:3], marker='o')
    for j in range(m):
        plt.text(j * 5, history[j * 5, 1], f"({j * 5}, {history[j * 5, 1]:.2f})",
                 horizontalalignment='left', verticalalignment='bottom')
    for j in range(m):
        plt.text(j * 5, history[j * 5, 2], f"({j * 5}, {history[j * 5, 2]:.2f})",
                 horizontalalignment='left', verticalalignment='bottom')
    plt.legend(['Tr Accuracy', 'All Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.xticks(np.arange(0, num_epochs, 2))  # 设置横坐标刻度为2的倍数
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig(save_dir + dt + '_accuracy_curve_task_' + str(i + 1) + '.png')
    plt.show()
    plt.clf()

    columns = ['Tr_loss', 'Tr_acc', 'AV_acc']
    df = pd.DataFrame(history)
    df.columns = columns

    # 保存csv和xlsx结果
    df.to_csv(save_dir + dt + 'output_task_' + str(i + 1) + '.csv', index=False)
    df = pd.DataFrame(history, columns=['loss', 'train_acc', 'test_acc'])
    df.to_excel(save_dir + 'output' + str(i + 1) + '.xlsx', index=False)
