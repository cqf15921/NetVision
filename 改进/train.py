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
    processed_dir = "data/processed"

    # 【修复】同时加载训练集和测试集
    train_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = train_dataset.get_num_classes()
    print(f"[*] 检测到类别数: {num_classes}, 训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

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
        # ====================
        # [A] 训练阶段
        # ====================
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{args.epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Train Loss: {loss.item():.4f}, Train Acc: {100. * correct_train / total_train:.2f}%')

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        train_acc = 100. * correct_train / total_train
        train_loss_avg = running_loss / len(train_loader)

        # ====================
        # [B] 验证阶段 【修复增加的逻辑】
        # ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(test_loader)

        print(f'==> Epoch {epoch + 1} 总结: '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')

        # ====================
        # [C] 保存最佳模型 【修正为按 Val Acc 保存】
        # ====================
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'checkpoints/netvision_{safe_dataset_name}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'    [+] 发现最佳模型 (Val Acc: {best_acc:.2f}%)，已更新并保存至: {save_path}')

    print(f"[*] 训练结束! 最佳测试集准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    train()