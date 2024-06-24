import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def split_data(data_dir):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为张量
    ])

    # 下载并加载 CIFAR-100 数据集
    train_dataset = torchvision.datasets.CIFAR100(root=data_dir,
                                                  train=True,
                                                  download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_dir,
                                                 train=False,
                                                 download=True, transform=transform)

    # 将训练集、数据集的标签转换为 numpy 数组
    train_targets = np.array(train_dataset.targets)
    train_data = np.array(train_dataset.data)
    train_data = train_data.transpose(0, 3, 1, 2)

    test_targets = np.array(test_dataset.targets)
    test_data = np.array(test_dataset.data)
    test_data = test_data.transpose(0, 3, 1, 2)

    # 划分基础集（类别 0 到 49）
    initial_class_num = 10
    initial_classes = range(initial_class_num)

    initial_train_idx = np.isin(train_targets, initial_classes)
    initial_train_data = train_data[initial_train_idx]
    initial_train_target = train_targets[initial_train_idx]

    initial_test_idx = np.isin(test_targets, initial_classes)
    initial_test_data = test_data[initial_test_idx]
    initial_test_target = test_targets[initial_test_idx]

    incre_task_num = 9
    incre_class_num = 10
    incremental_train_data = []
    incremental_train_target = []
    incremental_test_data = []
    incremental_test_target = []

    # 划分增量集（每个增量集包含 10 个类别）
    for i in range(incre_task_num):
        incremental_train_classes = range(initial_class_num + i * incre_class_num,
                                          initial_class_num + (i + 1) * incre_class_num)
        incremental_test_classes = range(0, initial_class_num + (i + 1) * incre_class_num)

        inc_train_idx = np.isin(train_targets, incremental_train_classes)
        inc_train_data = train_data[inc_train_idx]
        inc_train_target = train_targets[inc_train_idx]

        inc_test_idx = np.isin(test_targets, incremental_test_classes)
        inc_test_data = test_data[inc_test_idx]
        inc_test_target = test_targets[inc_test_idx]

        incremental_train_data.append(inc_train_data)
        incremental_train_target.append(inc_train_target)
        incremental_test_data.append(inc_test_data)
        incremental_test_target.append(inc_test_target)

    # initial_train_data, initial_train_target 是基础训练集的数据和标签
    # initial_test_data, initial_test_target 是基础测试集的数据和标签
    # incremental_train_data, incremental_train_target 是列表，每个元素对应一个增量训练集的数据和标签
    # incremental_test_data, incremental_test_target 是列表，每个元素对应一个增量测试集的数据和标签

    """
    def print_unique_labels(data_targets, dataset_name):
        unique_labels = set(data_targets)
        print(f"{dataset_name} unique labels:")
        for label in sorted(unique_labels):
            print(label)
        print()


    # 打印基础训练集和测试集的唯一标签
    print_unique_labels(initial_train_target, "Initial Train")
    print_unique_labels(initial_test_target, "Initial Test")

    # 打印每个增量训练集和测试集的唯一标签
    for i in range(incre_task_num):
        print_unique_labels(incremental_train_target[i], f"Incremental Train {i + 1}")
        print_unique_labels(incremental_test_target[i], f"Incremental Test {i + 1}")
    """
    return (initial_train_data, initial_train_target, initial_test_data, initial_test_target,
            incremental_train_data, incremental_train_target, incremental_test_data, incremental_test_target)
