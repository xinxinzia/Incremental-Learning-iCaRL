import logging
import numpy as np
import shutil
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from csv_func import preserve_csv
from csv_func import read_csv
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import datetime
from torchvision.transforms import ToTensor
from PIL import Image
import data_split1
import data_split2
import data_split3
import random


def extract_dataset(i, base_classes, incre_classes, num_extract, data_dir, istype, old_merged_train_target,
                    old_merged_train_data):
    if istype == 1:
        (initial_train_data, initial_train_target, initial_test_data, initial_test_target,
         incremental_train_data, incremental_train_target, incremental_test_data,
         incremental_test_target) = data_split1.split_data(data_dir)
    elif istype == 2:
        (initial_train_data, initial_train_target, initial_test_data, initial_test_target,
         incremental_train_data, incremental_train_target, incremental_test_data,
         incremental_test_target) = data_split2.split_data(data_dir)
    elif istype == 3:
        (initial_train_data, initial_train_target, initial_test_data, initial_test_target,
         incremental_train_data, incremental_train_target, incremental_test_data,
         incremental_test_target) = data_split3.split_data(data_dir)

    if i == 0:
        train_data = initial_train_data
        train_targets = initial_train_target
    else:
        train_data = incremental_train_data[i - 1]
        train_targets = incremental_train_target[i - 1]

    # train_targets, train_data = old_merged_train_target, old_merged_train_data

    new_train_data = incremental_train_data[i]
    new_train_targets = incremental_train_target[i]

    prev_classes = range(base_classes + i * incre_classes)
    prev_train_data = []
    prev_train_target = []

    # Extract features and select samples randomly
    for class_idx in prev_classes:
        class_indices = [idx for idx in range(len(train_data)) if train_targets[idx] == class_idx]
        selected_indices = random.sample(class_indices, min(num_extract, len(class_indices)))

        prev_train_data.extend([train_data[idx] for idx in selected_indices])
        prev_train_target.extend([class_idx] * len(selected_indices))

    prev_train_data = np.array(prev_train_data)
    prev_train_target = np.array(prev_train_target)
    if i > 0:
        prev_train_data = np.concatenate((prev_train_data, old_merged_train_data))
        prev_train_target = np.concatenate((prev_train_target, old_merged_train_target))

    # Merge with new incremental classes
    new_classes = range(base_classes + i * incre_classes, base_classes + (i + 1) * incre_classes)
    merged_train_data = np.concatenate((prev_train_data, new_train_data))
    merged_train_target = np.concatenate((prev_train_target, new_train_targets))

    return merged_train_target, merged_train_data

# def extract_dataset(i, incre_classes, num_extract, data_dir, resnet50, transform_test):
#     (initial_train_data, initial_train_target, initial_test_data, initial_test_target,
#      incremental_train_data, incremental_train_target, incremental_test_data,
#      incremental_test_target) = data_split1.split_data(data_dir)
#
#     if i == 0:
#         train_data = initial_train_data
#         train_targets = initial_train_target
#     else:
#         train_data = incremental_train_data[i - 1]
#         train_targets = incremental_train_target[i - 1]
#
#     prev_classes = range(50 + i * incre_classes)  # Adjust based on your specific logic
#     prev_train_data = []
#     prev_train_target = []
#
#     # Extract features and select samples
#     for class_idx in prev_classes:
#         features_list = []
#
#         for idx in range(len(train_data)):
#             if train_targets[idx] == class_idx:
#                 img_path = train_data[idx]  # Assuming train_data contains paths to images
#                 img = Image.fromarray(img_path.astype('uint8')).convert('RGB')
#                 img = transform_test(img)
#                 input_batch = img.unsqueeze(0).to('cuda:0')
#
#                 # Extract features (excluding final layer)
#                 with torch.no_grad():
#                     features = resnet50(input_batch)
#
#                 features = features.squeeze()
#                 features_list.append(features)
#
#         if features_list:
#             features = torch.stack(features_list)
#             mean_feature = torch.mean(features, dim=0)
#             distances = torch.norm(features - mean_feature, dim=1)
#             selected_indices = np.argsort(distances.cpu().numpy())[:num_extract]
#
#             prev_train_data.extend([train_data[idx] for idx in selected_indices])
#             prev_train_target.extend([class_idx] * len(selected_indices))
#
#     prev_train_data = np.array(prev_train_data)
#     prev_train_target = np.array(prev_train_target)
#
#     # Merge with new incremental classes
#     new_classes = range(50 + i * incre_classes, 50 + (i + 1) * incre_classes)
#     merged_train_data = np.concatenate((prev_train_data, train_data))
#     merged_train_target = np.concatenate((prev_train_target, train_targets))
#
#     return merged_train_target, merged_train_data
