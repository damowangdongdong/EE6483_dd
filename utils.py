import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from tqdm import tqdm
import csv
import os
from datetime import datetime
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root, extensions, transform=None):
        self.root = root
        self.extensions = extensions
        self.transform = transform
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in sorted(fnames):
                if fname.lower().endswith(self.extensions):
                    path = os.path.join(root, fname)
                    samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loss_function(num_classes):
    if num_classes == 2:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        if isinstance(loss_fn, nn.BCEWithLogitsLoss):
            targets = targets.unsqueeze(1).float()  # 调整目标尺寸并转换为浮点类型
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, loss_fn, device, num_classes):
    model.eval()
    total_loss = 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                targets = targets.unsqueeze(1).float()  # 调整目标尺寸并转换为浮点类型
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            if num_classes == 2:
                predictions = torch.sigmoid(outputs).round()
            else:
                predictions = torch.argmax(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    return total_loss / len(dataloader), accuracy, f1

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def test_model(model, dataloader, device, is_custom=False):
    model.eval()
    all_predictions = []
    all_targets = []
    correct_images = []
    incorrect_images = []
    image_names = []
    correct_outputs = []
    incorrect_outputs = []
    correct_predictions = []
    incorrect_predictions = []
    custom_predictions = []

    if is_custom:
        # 定义转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # 其他转换
        ])
        # 使用 DatasetFolder 并指定加载函数
        custom_dataset = CustomDataset(root='./datasets/test', extensions=('jpg', 'jpeg', 'png'), transform=transform)
        custom_dataloader = DataLoader(custom_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            if is_custom:
                correct_mask = targets == predictions
                incorrect_mask = targets != predictions
                correct_images.extend(inputs[correct_mask].cpu())
                incorrect_images.extend(inputs[incorrect_mask].cpu())
                correct_outputs.extend(outputs[correct_mask].cpu())
                incorrect_outputs.extend(outputs[incorrect_mask].cpu())
                correct_predictions.extend(predictions[correct_mask].cpu())
                incorrect_predictions.extend(predictions[incorrect_mask].cpu())
                image_names.extend(dataloader.dataset.samples)  # 假设 dataloader.dataset.samples 包含图像名称

        if is_custom:
            for inputs in tqdm(custom_dataloader, desc="Custom Testing"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions = torch.sigmoid(outputs).round()
                custom_predictions.extend(predictions.cpu().numpy())

            # 保存预测结果到 CSV 文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f'custom_predictions_{timestamp}.csv'
            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Image', 'Prediction'])
                for i, (path, prediction) in enumerate(zip(custom_dataset.samples, custom_predictions)):
                    writer.writerow([path[0], prediction])

    if is_custom:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f'predictions_{timestamp}.pdf'
        pdf = FPDF()
        
        # 仅保存两个正确判别和两个错误判别的图像
        for i in range(min(2, len(correct_images))):
            pdf.add_page()
            img = correct_images[i]
            output = correct_outputs[i]
            prediction = correct_predictions[i]
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到 0 到 1 范围
            img_path = f"correct_{i}_{timestamp}.png"
            plt.imsave(img_path, img)
            pdf.image(img_path, x=10, y=10, w=100)
            pdf.set_xy(10, 120)
            pdf.set_font("Arial", size=12)
            output_list = output.numpy().tolist()
            output_str = ', '.join([f'{val:.4f}' for val in output_list])
            pdf.cell(200, 10, f"Output: [{output_str}], Prediction: {prediction.numpy()}", ln=True)
            os.remove(img_path)

        for i in range(min(2, len(incorrect_images))):
            pdf.add_page()
            img = incorrect_images[i]
            output = incorrect_outputs[i]
            prediction = incorrect_predictions[i]
            img = img.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # 归一化到 0 到 1 范围
            img_path = f"incorrect_{i}_{timestamp}.png"
            plt.imsave(img_path, img)
            pdf.image(img_path, x=10, y=10, w=100)
            pdf.set_xy(10, 120)
            pdf.set_font("Arial", size=12)
            output_list = output.numpy().tolist()
            output_str = ', '.join([f'{val:.4f}' for val in output_list])
            pdf.cell(200, 10, f"Output: [{output_str}], Prediction: {prediction.numpy()}", ln=True)
            os.remove(img_path)

        pdf.output(pdf_filename)
    else:
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        print(f"Test Accuracy: {accuracy}")
        print(f"Test F1 Score: {f1}")

    return all_predictions, correct_images, incorrect_images

def save_images_to_pdf(correct_images, incorrect_images, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.pdf"
    pdf = FPDF()
    for img in correct_images + incorrect_images:
        pdf.add_page()
        pdf.image(img, x=10, y=10, w=100)
    pdf.output(filename)



def mean_and_95ci(data):
    mean = np.mean(data)
    ci95 = 1.96 * np.std(data) / np.sqrt(len(data))
    return mean, ci95

def plot_and_save_confusion_matrix(labels, predictions, class_names, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"confusion_matrix_{timestamp}.png"
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)


def train_and_evaluate(model, train_loader, val_loader, test_loader, num_classes, device, epochs, optimizer_fn, lr, lr_decay, patience, is_custom=False):
    optimizer = optimizer_fn(model.parameters(), lr=lr)
    loss_fn = get_loss_function(num_classes)
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_accuracy, val_f1 = validate_one_epoch(model, val_loader, loss_fn, device, num_classes)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}, Val F1: {val_f1}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay['gamma']  # 使用 lr_decay['gamma']

    model.load_state_dict(best_model)
    test_predictions, correct_images, incorrect_images = test_model(model, test_loader, device, is_custom)
    return test_predictions, correct_images, incorrect_images, val_accuracy, val_f1

def train_and_evaluate_with_seeds(model_fn, train_loader, val_loader, test_loader, num_classes, device, epochs, optimizer_fn, lr, lr_decay, patience, is_custom, seeds, class_names):
    all_accuracies = []
    all_f1_scores = []

    if isinstance(seeds, int):
        seeds = [seeds]

    for seed in seeds:
        set_random_seed(seed)
        model = model_fn().to(device)
        test_predictions, correct_images, incorrect_images, val_accuracy, val_f1 = train_and_evaluate(model, train_loader, val_loader, test_loader, num_classes, device, epochs, optimizer_fn, lr, lr_decay, patience, is_custom)
        
        if is_custom:
            accuracy = val_accuracy
            f1 = val_f1
        else:
            accuracy = accuracy_score(test_loader.dataset.targets, test_predictions)
            f1 = f1_score(test_loader.dataset.targets, test_predictions, average='weighted')
        
        all_accuracies.append(accuracy)
        all_f1_scores.append(f1)

    mean_accuracy, ci95_accuracy = mean_and_95ci(all_accuracies)
    mean_f1, ci95_f1 = mean_and_95ci(all_f1_scores)
    plot_and_save_confusion_matrix(test_loader.dataset.targets, test_predictions, class_names)
    return mean_accuracy, ci95_accuracy, mean_f1, ci95_f1
