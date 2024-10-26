import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
from sklearn.metrics import accuracy_score

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using device: {device}')

# 开关控制是否重新训练模型
RETRAIN_MODEL = True  # 设置为 True 则重新训练模型，设置为 False 则加载已保存的模型

class CustomDataset(Dataset):
    def __init__(self, image_dir, xml_path, transform=None):
        self.image_dir = image_dir
        self.ratings = self.load_ratings(xml_path)
        self.image_files = list(self.ratings.keys())
        self.labels = [int(score) for score in self.ratings.values()]
        self.transform = transform
        self.label_counts = self.get_label_counts()

        # 计算每个标签的均衡样本数
        self.target_count = 750
        self.balanced_data = self.balance_samples()
        # self.print_label_counts()  # 打印标签数量

    def load_ratings(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ratings = {}
        for img in root.findall('image'):
            name = img.get('name')
            score = float(img.find('average_score').text)
            ratings[name] = score
        return ratings

    def get_label_counts(self):
        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def balance_samples(self):
        balanced_data = []
        for label, count in self.label_counts.items():
            indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
            if count < self.target_count:
                # 数据增强
                balanced_data.extend(indices)  # 添加原始样本
                temp = self.target_count - len(indices)
                while temp > 0:
                    # 随机选择一张原始样本进行增强
                    balanced_data.append(random.choice(indices))
                    temp -= 1
            else:
                # 下采样
                balanced_data.extend(random.sample(indices, self.target_count))
        return balanced_data

    def __len__(self):
        return len(self.balanced_data)

    def __getitem__(self, idx):
        img_idx = self.balanced_data[idx]
        img_file = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, img_file)
        img = Image.open(img_path).convert('L')  # 确保为单通道
        # img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        label = self.labels[img_idx]
        return img, label

    def print_label_counts(self):
        # 计算每个标签的数量
        label_indices = [self.labels[i] for i in self.balanced_data]
        count = Counter(label_indices)
        print("每个标签的样本数量:")
        for label, qty in count.items():
            print(f"标签 {label}: {qty}")


# 设置数据路径
data_dir = 'train_data'
xml_path = 'average_scores.xml'

# 数据增强和预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_features = self.model.fc.in_features  # Get the number of input features for the last layer
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5)  # Output layer for your classification task
        )

    def forward(self, x):
        x = self.model(x)
        return x

def train_and_validate(model, train_loader, test_loader, epochs):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (predictions == labels.view(-1).long()).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples * 100
        train_losses.append(train_loss)

        # 验证模型
        model.eval()
        total_val_loss = 0
        total_val_correct = 0
        total_val_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels.long())

                total_val_loss += loss.item()
                total_val_correct += (predictions == labels.view(-1).long()).sum().item()
                total_val_samples += labels.size(0)

        val_loss = total_val_loss / len(test_loader)
        val_accuracy = total_val_correct / total_val_samples * 100
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    return train_losses, val_losses

# 训练模型
if RETRAIN_MODEL:
    # 创建数据集和数据加载器
    dataset = CustomDataset(data_dir, xml_path, transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 训练过程
    # 冻结所有层，除了最后一层
    for param in model.model.parameters():
        param.requires_grad = False

    # 仅对最后一层的参数进行训练
    for param in model.model.fc.parameters():
        param.requires_grad = True

    # 训练模型（前 10 个 epochs）
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, epochs=10)

    # 解冻所有层
    for param in model.model.parameters():
        param.requires_grad = True

    # 重新定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 继续训练（后 20 个 epochs）
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, epochs=50)

    # 绘制损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 保存模型
    torch.save(model.state_dict(), 'facial_attractiveness_model.pth')
else:
    model = ResNet().to(device)
    model.load_state_dict(torch.load('facial_attractiveness_model.pth'))
    model.eval()

    # 设置预测数据路径
    predict_dir = 'val_data'
    # 创建 XML 根节点
    results = ET.Element("predictions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for img_file in os.listdir(predict_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(predict_dir, img_file)
                img = Image.open(img_path).convert('L')  # 确保为单通道
                img = transform(img).unsqueeze(0).to(device)

                output = model(img)
                predicted_score = torch.argmax(output, dim=1).item()

                print(f'Predicted score for {img_file}: {predicted_score}')

                # 添加到 XML
                image_element = ET.SubElement(results, "image", name=img_file)
                ET.SubElement(image_element, "predicted_score").text = str(predicted_score)

    # 保存预测结果为 XML 文件
    tree = ET.ElementTree(results)
    tree.write("predict_scores.xml", encoding='utf-8', xml_declaration=True)  # 保存为 predict_scores.xml

