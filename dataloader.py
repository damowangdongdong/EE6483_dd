import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

def get_transforms(augmentations, use_normalize, imgsize, crop_type):
    transform_list = []
    transform_list.append(transforms.Resize((imgsize, imgsize)))
    if crop_type == 'random':
        transform_list.append(transforms.RandomCrop(imgsize))
    elif crop_type == 'center':
        transform_list.append(transforms.CenterCrop(imgsize))
    if 'random_horizontal_flip' in augmentations:
        transform_list.append(transforms.RandomHorizontalFlip())
    if 'auto_augment' in augmentations:
        transform_list.append(transforms.AutoAugment())
    if 'rand_augment' in augmentations:
        transform_list.append(transforms.RandAugment())
    transform_list.append(transforms.ToTensor())
    if use_normalize:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    device = x.device
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    '''Returns cutmixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    device = x.device
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)  # 修改为 int
    cut_h = int(H * cut_rat)  # 修改为 int

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def load_dataset(dataset_name, transform):
    if dataset_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        train_dataset = datasets.ImageFolder(root='./datasets/train', transform=transform)
        val_dataset = datasets.ImageFolder(root='./datasets/val', transform=transform)
        test_dataset = datasets.ImageFolder(root='./datasets/val', transform=transform)
    return train_dataset, val_dataset, test_dataset

def print_and_save_dataset_info(dataset, dataset_name, phase):
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    info = f"{phase} dataset ({dataset_name}):\nTotal samples: {len(dataset)}\nClass distribution: {class_counts}\n"
    print(info)
    with open(f"{dataset_name}_{phase}_dataset_info.txt", "w") as f:
        f.write(info)

def create_dataloader(augmentations, dataset_name, use_normalize, imgsize, crop_type, batch_size=32, use_mixup=False, use_cutmix=False):
    transform = get_transforms(augmentations, use_normalize, imgsize, crop_type)
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if use_mixup:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
            # 使用混合后的数据进行训练
            # 例如：outputs = model(inputs)
            # loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
    
    if use_cutmix:
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets)
            # 使用混合后的数据进行训练
            # 例如：outputs = model(inputs)
            # loss = cutmix_criterion(criterion, outputs, targets_a, targets_b, lam)
    
    print_and_save_dataset_info(train_dataset, dataset_name, "train")
    print_and_save_dataset_info(val_dataset, dataset_name, "val")
    print_and_save_dataset_info(test_dataset, dataset_name, "test")
    
    return train_loader, val_loader, test_loader

# 示例用法
# augmentations = ['resize', 'random_crop', 'random_horizontal_flip', 'auto_augment', 'rand_augment']
# dataset_name = 'custom'  # 或 'CIFAR10'
# use_normalize = True
# imgsize = 128
# crop_type = 'random'  # 或 'center'
# batch_size = 32
# use_mixup = True
# use_cutmix = True
# train_loader, val_loader, test_loader = create_dataloader(augmentations, dataset_name, use_normalize, imgsize, crop_type, batch_size, use_mixup, use_cutmix)

# 假设 outputs 是模型输出，labels 是目标张量
