import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os

import pandas as pd

from tqdm import tqdm

import os

import datetime
from extract_dataset import extract_dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 让程序报错


# 定义Label Smoothing函数
def label_smoothing(loss_function, outputs, labels):
    nll_loss = loss_function(outputs, labels)
    smoothing = 0.1
    confidence = 1.0 - smoothing
    logprobs = nn.functional.log_softmax(outputs.data, dim=-1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()
# def label_smoothing(outputs, labels, num_classes, smoothing=0.1):
#     # 使用交叉熵损失计算负对数似然损失
#     nll_loss = F.cross_entropy(outputs, labels)
#     # 计算 log 概率
#     logprobs = F.log_softmax(outputs, dim=-1)
#     # 转换标签为 one-hot 编码
#     targets = F.one_hot(labels, num_classes).float()
#     # 应用标签平滑
#     targets = (1 - smoothing) * targets + smoothing / num_classes
#     # 计算标签平滑后的损失
#     smooth_loss = -(targets * logprobs).sum(dim=-1)
#     # 最终损失是原始损失和标签平滑损失的加权和
#     loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss.mean()
#     return loss


# 定义知识蒸馏算法函数
def distillation_loss(outputs, ls_loss, teacher_outputs):
    t = 7  # 蒸馏温度
    alpha = 0.3
    soft_loss = nn.KLDivLoss(reduction="batchmean")  # KL散度损失函数
    # reduction：指定损失输出的形式，batchmean：将输出的总和除以batchsize
    zz_loss = soft_loss(nn.functional.log_softmax(outputs / t, dim=1),
                        nn.functional.softmax(teacher_outputs / t, dim=1))
    loss = ls_loss * alpha + zz_loss * (1 - alpha)
    # 新数据分类loss和旧数据蒸馏loss组成新的loss
    return loss


# 定义训练函数
def train_and_valid(model, train_data, val_data, device, loss_function, optimizer, epochs, i, train_data_size, valid_data_size):
    # 初始化参数
    history = []  # 存储训练和验证过程中的损失值和准确率
    best_acc = 0.0  # 精确度
    best_epoch = 0  # 迭代次数
    best_model = model  # 初始化最佳模型

    for epoch in range(epochs):
        epoch_start = time.time()  # 每次迭代时间
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        # 训练和验证集的精确度和损失率
        train_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        for n, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(torch.float32).to(device)
            labels = labels.to(device)
            # inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            # labels = torch.tensor(labels, dtype=torch.long).to(device)

            # 因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
            outputs = model(inputs)
            if i > 0:
                model.eval()
                t_outputs = model(inputs)
            model.train()  # 将模型置于训练模式

            ls_loss = label_smoothing(loss_function, outputs, labels)  # 标签平滑
            if i > 0:
                loss = distillation_loss(outputs, ls_loss, t_outputs)  # 知识蒸馏算法
            else:
                loss = label_smoothing(loss_function, outputs, labels)
            loss.requires_grad_(True)
            loss.backward()  # 反向传播
            optimizer.step()  # 通过 step 方法更新优化器参数，以最小化损失函数值
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)  # predictions:最大预测值对应的标签
            correct_counts = predictions.eq(labels.data.view_as(predictions))  # 计算模型预测正确的布尔列表
            acc = torch.mean(correct_counts.type(torch.FloatTensor))  # 求正确率
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()  # 启用评估模式，不会改变dropout值
            # 计算验证集的损失值和正确率
            for j, (inputs, labels) in enumerate(tqdm(val_data)):
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        # 计算训练集和验证集的损失率和正确率
        avg_train_loss = train_loss / train_data_size

        avg_train_acc = train_acc / train_data_size
        avg_valid_acc = valid_acc / valid_data_size

        history.append([avg_train_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
            best_model = model  # 更新最佳模型参数

        epoch_end = time.time()

        print(
            "Task:{:03d}, Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tAll Validation: Accuracy:{:.4f}%  Time: {:.4f}s".format(
                i + 1, epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for current validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return best_model, history

# def train_and_valid(model, train_data, val_data, device, loss_function, optimizer, epochs, i, train_data_size, valid_data_size):
#     import time
#     from tqdm import tqdm
#     import torch
#
#     # 初始化参数
#     history = []  # 存储训练和验证过程中的损失值和准确率
#     best_acc = 0.0  # 精确度
#     best_epoch = 0  # 迭代次数
#     best_model = model  # 初始化最佳模型
#
#     # 获取模型输出的类数
#     try:
#         num_classes = model.fc.out_features  # 对于ResNet模型
#     except AttributeError:
#         raise ValueError("The model does not have an attribute 'fc'. Check if the model structure is correct.")
#
#     for epoch in range(epochs):
#         epoch_start = time.time()  # 每次迭代时间
#         print("Epoch: {}/{}".format(epoch + 1, epochs))
#
#         # 训练和验证集的精确度和损失率
#         train_loss = 0.0
#         train_acc = 0.0
#         valid_acc = 0.0
#
#         model.train()  # 将模型置于训练模式
#
#         for n, (inputs, labels) in enumerate(tqdm(train_data)):
#             inputs = inputs.to(torch.float32).to(device)
#             labels = labels.to(device)
#
#             # 数据检查
#             assert labels.max().item() < num_classes, f"Label value {labels.max().item()} out of range"
#             assert labels.min().item() >= 0, "Label value less than 0"
#
#             # 因为这里梯度是累加的，所以每次记得清零
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             if i > 0:
#                 model.eval()
#                 t_outputs = model(inputs)
#             model.train()  # 将模型置于训练模式
#
#             ls_loss = label_smoothing(loss_function, outputs, labels)  # 标签平滑
#             if i > 0:
#                 loss = distillation_loss(outputs, ls_loss, t_outputs)  # 知识蒸馏算法
#             else:
#                 loss = ls_loss
#             loss.requires_grad_(True)
#             loss.backward()  # 反向传播
#             optimizer.step()  # 通过 step 方法更新优化器参数，以最小化损失函数值
#             train_loss += loss.item() * inputs.size(0)
#             ret, predictions = torch.max(outputs.data, 1)  # predictions: 最大预测值对应的标签
#             correct_counts = predictions.eq(labels.data.view_as(predictions))  # 计算模型预测正确的布尔列表
#             acc = torch.mean(correct_counts.type(torch.FloatTensor))  # 求正确率
#             train_acc += acc.item() * inputs.size(0)
#
#         with torch.no_grad():
#             model.eval()  # 启用评估模式，不会改变dropout值
#             # 计算验证集的损失值和正确率
#             for j, (inputs, labels) in enumerate(tqdm(val_data)):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 # 数据检查
#                 assert labels.max().item() < num_classes, f"Label value {labels.max().item()} out of range"
#                 assert labels.min().item() >= 0, "Label value less than 0"
#
#                 outputs = model(inputs)
#                 ret, predictions = torch.max(outputs.data, 1)
#                 correct_counts = predictions.eq(labels.data.view_as(predictions))
#                 acc = torch.mean(correct_counts.type(torch.FloatTensor))
#                 valid_acc += acc.item() * inputs.size(0)
#
#         # 计算训练集和验证集的损失率和正确率
#         avg_train_loss = train_loss / train_data_size
#         avg_train_acc = train_acc / train_data_size
#         avg_valid_acc = valid_acc / valid_data_size
#
#         history.append([avg_train_loss, avg_train_acc, avg_valid_acc])
#
#         if best_acc < avg_valid_acc:
#             best_acc = avg_valid_acc
#             best_epoch = epoch + 1
#             best_model = model  # 更新最佳模型参数
#
#         epoch_end = time.time()
#         print(f'Epoch {epoch + 1} completed in {epoch_end - epoch_start:.2f} seconds')
#         print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {avg_train_acc:.4f}')
#         print(f'Validation Accuracy: {avg_valid_acc:.4f}')
#
#     return history, best_model, best_epoch, best_acc
