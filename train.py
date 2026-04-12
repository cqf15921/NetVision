import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.netvision_model import NetVision
from utils.dataset import NetVisionDataset


def train():
    parser = argparse.ArgumentParser(description='NetVision 深度学习模型训练脚本')
    # 核心参数：选择数据集
    parser.add_argument('--dataset', type=str, default='CIC_IoT_2023',
                        choices=['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT'],
                        help='选择要训练的数据集')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')

    # 这里是修正点：必须写 type=float
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')

    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

    args = parser.parse_args()

    # 1. 动态确定路径与类别数
    safe_dataset_name = args.dataset.lower().replace('-', '_')
    train_path = f'data/processed/{safe_dataset_name}_dataset_train.npz'

    if not os.path.exists(train_path):
        print(f"[!] 错误: 找不到预处理后的训练集文件 {train_path}，请先运行 preprocessing.py")
        return

    # 2. 加载数据集
    print(f"[*] 正在加载数据集: {args.dataset}...")
    train_dataset = NetVisionDataset(train_path)
    # 在 Colab 或 Linux 上 num_workers 建议设为 2 或 4 以加速读取
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    num_classes = len(train_dataset.classes)
    print(f"[*] 检测到类别数: {num_classes}, 样本总数: {len(train_dataset)}")

    # 3. 初始化模型
    model = NetVision(num_classes=num_classes).to(args.device)

    # 4. 优化策略：AdamW + CosineAnnealingLR
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print(f"[*] 开始在 {args.device} 上训练...")
    os.makedirs('checkpoints', exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Acc: {100. * correct / total:.2f}%')

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_acc = 100. * correct / total
        print(
            f'==> Epoch {epoch + 1} 完成! 平均 Loss: {running_loss / len(train_loader):.4f}, Acc: {epoch_acc:.2f}%, 当前 LR: {current_lr:.6f}')

        # 6. 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            save_path = f'checkpoints/netvision_{safe_dataset_name}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'[+] 最佳模型已更新并保存至: {save_path}')

    print(f"[*] 训练结束! 最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    train()