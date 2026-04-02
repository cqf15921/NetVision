import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.lightguard_model import LightGuard
from utils.dataset import TrafficDataset
import os


def train():
    # 1. 基本配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 15  # USTC_TFC2016 专用
    save_path = './checkpoints'
    os.makedirs(save_path, exist_ok=True)

    # 2. 加载数据集
    # 注意：这里分别加载 preprocessing.py 严格划分好的训练集和测试集
    # 我们将测试集（_test.npz）作为训练过程中的验证集（val_loader）来监控性能
    train_dataset = TrafficDataset("data/processed/ustc_tfc2016_dataset_train.npz")
    val_dataset = TrafficDataset("data/processed/ustc_tfc2016_dataset_test.npz")

    num_classes = train_dataset.get_num_classes()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. 初始化模型、损失函数和优化器
    model = LightGuard().to(device)

    # 动态调整输出层以匹配 USTC_TFC2016 的 20 个类别
    model.f1[2] = nn.Linear(256, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

    writer = SummaryWriter('./logs/lightguard_train')

    # 4. 训练循环
    print(f"开始训练，设备: {device}, 类别数: {num_classes}")
    print(f"训练集样本数: {len(train_dataset)}, 验证/测试集样本数: {len(val_dataset)}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

        # 记录日志
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 保存最新/最佳模型
        torch.save(model.state_dict(), os.path.join(save_path, 'lightguard_ustc.pth'))

    writer.close()
    print("训练完成！模型权重已保存至 checkpoints 目录。")


if __name__ == "__main__":
    train()