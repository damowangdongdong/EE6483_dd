import argparse
import yaml
import torch
import torch.optim as optim
import torchvision.models as models
import timm
from dataloader import create_dataloader
from utils import train_and_evaluate_with_seeds

def get_model(model_name, num_classes=2, pretrained=False):
    if model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        if num_classes == 2:
            model.fc = torch.nn.Linear(model.fc.in_features, 1)  # 修改最后一层为单一神经元
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50_untrained':
        model = models.resnet50(weights=None)
        if num_classes == 2:
            model.fc = torch.nn.Linear(model.fc.in_features, 1)  # 修改最后一层为单一神经元
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vit_b':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        if num_classes == 2:
            model.head = torch.nn.Linear(model.head.in_features, 1)  # 修改最后一层为单一神经元
        else:
            model.head = torch.nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'convnextv2':
        model = timm.create_model('convnextv2_base', pretrained=pretrained)
        if num_classes == 2:
            model.head = torch.nn.Linear(model.head.in_features, 1)  # 修改最后一层为单一神经元
        else:
            model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model
