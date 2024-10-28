import argparse
import yaml
import torch
import torch.optim as optim
from dataloader import create_dataloader
from model import get_model
from utils import train_and_evaluate_with_seeds
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run training and evaluation")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--epochs', type=int, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, help="Batch size for training")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--optimizer', type=str, help="Optimizer to use")
    parser.add_argument('--patience', type=int, help="Patience for early stopping")
    parser.add_argument('--seeds', type=int, nargs='+', help="Random seeds for training")
    parser.add_argument('--augmentations', type=str, nargs='+', help="List of augmentations to apply")
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset")
    parser.add_argument('--use_normalize', type=bool, help="Whether to use normalization")
    parser.add_argument('--imgsize', type=int, help="Image size")
    parser.add_argument('--crop_type', type=str, help="Type of crop (random or center)")
    parser.add_argument('--use_mixup', type=bool, help="Whether to use mixup")
    parser.add_argument('--use_cutmix', type=bool, help="Whether to use cutmix")
    parser.add_argument('--num_classes', type=int, help="Number of classes")
    parser.add_argument('--class_names', type=str, nargs='+', help="List of class names")
    parser.add_argument('--pretrained', type=bool, help="Whether to use a pretrained model")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_optimizer_fn(optimizer_name):
    if optimizer_name == "Adam":
        return optim.Adam
    elif optimizer_name == "SGD":
        return optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def save_log(log_dir, config, model_name, results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.txt")
    os.makedirs(log_dir, exist_ok=True)
    
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"Model: {model_name}\n")
        log_file.write("Config:\n")
        yaml.dump(config, log_file)
        log_file.write("\nResults:\n")
        log_file.write(f"Mean Accuracy: {results['mean_accuracy']}\n")
        log_file.write(f"95% CI Accuracy: {results['ci95_accuracy']}\n")
        log_file.write(f"Mean F1 Score: {results['mean_f1']}\n")
        log_file.write(f"95% CI F1 Score: {results['ci95_f1']}\n")

def main():
    args = parse_args()
    config = load_config(args.config)

    # 覆盖配置文件中的参数
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['dataset']['batch_size'] = args.batch_size
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.optimizer is not None:
        config['training']['optimizer'] = args.optimizer
    if args.patience is not None:
        config['training']['patience'] = args.patience
    if args.seeds is not None:
        config['training']['seeds'] = args.seeds
    if args.augmentations is not None:
        config['dataset']['augmentations'] = args.augmentations
    if args.dataset_name is not None:
        config['dataset']['name'] = args.dataset_name
    if args.use_normalize is not None:
        config['dataset']['use_normalize'] = args.use_normalize
    if args.imgsize is not None:
        config['dataset']['imgsize'] = args.imgsize
    if args.crop_type is not None:
        config['dataset']['crop_type'] = args.crop_type
    if args.use_mixup is not None:
        config['dataset']['use_mixup'] = args.use_mixup
    if args.use_cutmix is not None:
        config['dataset']['use_cutmix'] = args.use_cutmix
    if args.num_classes is not None:
        config['evaluation']['num_classes'] = args.num_classes
    if args.class_names is not None:
        config['evaluation']['class_names'] = args.class_names
    if args.pretrained is not None:
        config['model']['pretrained'] = args.pretrained

    # 数据加载
    train_loader, val_loader, test_loader = create_dataloader(
        augmentations=config['dataset']['augmentations'],
        dataset_name=config['dataset']['name'],
        use_normalize=config['dataset']['use_normalize'],
        imgsize=config['dataset']['imgsize'],
        crop_type=config['dataset']['crop_type'],
        batch_size=config['dataset']['batch_size'],
        use_mixup=config['dataset']['use_mixup'],
        use_cutmix=config['dataset']['use_cutmix']
    )

    # 模型初始化
    model_fn = lambda: get_model(config['model']['name'], num_classes=config['evaluation']['num_classes'], pretrained=config['model']['pretrained'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 优化器
    optimizer_fn = get_optimizer_fn(config['training']['optimizer'])

    # 训练和评估
    mean_accuracy, ci95_accuracy, mean_f1, ci95_f1 = train_and_evaluate_with_seeds(
        model_fn=model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=config['evaluation']['num_classes'],
        device=device,
        epochs=config['training']['epochs'],
        optimizer_fn=optimizer_fn,
        lr=config['training']['lr'],
        lr_decay=config['training']['lr_decay'],
        patience=config['training']['patience'],  # 添加 patience 参数
        is_custom=(config['dataset']['name'] == 'custom'),
        seeds=config['training']['seeds'],
        class_names=config['evaluation']['class_names']
    )

    # 保存日志
    results = {
        'mean_accuracy': mean_accuracy,
        'ci95_accuracy': ci95_accuracy,
        'mean_f1': mean_f1,
        'ci95_f1': ci95_f1
    }
    save_log(log_dir="logs", config=config, model_name=config['model']['name'], results=results)

if __name__ == "__main__":
    main()