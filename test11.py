import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# 设置参数
BLOCK_SIZE = 2  # 2x2的小像素块
IMG_SIZE = 128  # 图像尺寸（128x128）
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

# 1. 图像预处理和NPR特征提取
def calculate_npr(image, block_size=BLOCK_SIZE):
    """计算图像的NPR特征"""
    h, w = image.shape[:2]
    npr_features = []

    # 将图像分块处理
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                # 计算像素差值
                diff = block - block[0, 0]  # 相对于左上角像素的差值
                npr_features.append(diff.flatten())

    # 将NPR特征展平
    npr_features = np.array(npr_features).flatten()
    return npr_features

# 2. 自定义数据集类
class FakeImageDataset(Dataset):
    def __init__(self, real_images, fake_images, transform=None):
        self.real_images = real_images
        self.fake_images = fake_images
        self.transform = transform

    def __len__(self):
        return len(self.real_images) + len(self.fake_images)

    def __getitem__(self, idx):
        if idx < len(self.real_images):
            image = self.real_images[idx]
            label = 0  # 真实图像标签
        else:
            image = self.fake_images[idx - len(self.real_images)]
            label = 1  # 伪造图像标签

        # 转换图像并计算NPR特征
        if self.transform:
            image = self.transform(image)
        npr_features = calculate_npr(image.numpy().squeeze())
        npr_features = torch.tensor(npr_features, dtype=torch.float32)

        return npr_features, label

# 3. 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 输出2类（真实/伪造）

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. 数据准备和训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=EPOCHS):
    loss_history = []  # 用于记录每个epoch的损失
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # inputs, labels = inputs.cuda(), labels.cuda()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_history.append(epoch_loss)  # 记录当前epoch的损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

    return loss_history  # 返回损失历史记录

# 5. 绘制损失曲线
def plot_loss(loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.xticks(range(1, len(loss_history) + 1))  # 设置 x 轴刻度
    plt.show()

# 6. 加载示例数据并训练模型
if __name__ == "__main__":
    # 生成一些示例数据 (用随机数据模拟)
    real_images = [np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8) for _ in range(100)]
    fake_images = [np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8) for _ in range(100)]

    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图像
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # 数据集和数据加载器
    dataset = FakeImageDataset(real_images, fake_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、损失函数和优化器
    input_size = (IMG_SIZE // BLOCK_SIZE) ** 2 * BLOCK_SIZE ** 2
    model = SimpleCNN(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 训练模型并获取损失历史
    loss_history = train_model(model, dataloader, criterion, optimizer)

    # 绘制损失曲线
    plot_loss(loss_history)
