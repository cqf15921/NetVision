import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 导入你的模型和数据集加载类 (已更新为 NetVision)
from models.netvision_model import NetVision
from utils.dataset import NetVisionDataset


def main():
    # ==========================================
    # 1. 解析命令行参数
    # ==========================================
    parser = argparse.ArgumentParser(description="NetVision 流量分类模型训练脚本")

    # 核心参数：选择数据集 (已更新为完全对齐论文的 PCAP 数据集)
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'ENTA_Datase', 'ToN-IoT'],
                        help='选择要训练的数据集')

    # 超参数配置
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

    args = parser.parse_args()

    print(f"=== 初始化 NetVision 训练 ===")
    print(f"[*] 目标数据集: {args.dataset}")
    print(f"[*] 训练设备:   {args.device.upper()}")
    print(f"[*] 超参数:     Epochs={args.epochs}, Batch_Size={args.batch_size}, LR={args.lr}")

    # ==========================================
    # 2. 准备数据和目录
    # ==========================================
    processed_dir = "data/processed"
    checkpoint_dir = "checkpoints"

    # 将可能存在的连字符(如ToN-IoT)替换为下划线，防止命名不规范
    safe_dataset_name = args.dataset.lower().replace('-', '_')

    # 日志目录更新为 netvision 命名前缀
    log_dir = f"logs/netvision_{safe_dataset_name}_{int(time.time())}"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 实例化 Dataset
    print("[*] 正在加载数据集...")
    try:
        train_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=True)
        test_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=False)
    except FileNotFoundError as e:
        print(f"[!] 报错: {e}")
        print("[!] 请先运行 utils/preprocessing.py 生成对应的 .npz 数据文件！")
        return

    # num_workers=4 可以加速数据加载，如果你在 Windows 下报错，可以改为 num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 动态获取当前数据集的类别数量
    num_classes = train_dataset.get_num_classes()
    print(f"[+] 数据集加载完成！训练集大小: {len(train_dataset)}, 类别数: {num_classes}")

    # ==========================================
    # 3. 初始化模型、损失函数和优化器
    # ==========================================
    model = NetVision().to(args.device)

    # 动态修改最后一层 (f1 的第3个元素) 以适配不同数据集的类别数量
    model.f1[2] = nn.Linear(256, num_classes).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # ==========================================
    # 4. 训练与验证循环
    # ==========================================
    best_acc = 0.0

    # 权重保存路径更新为 netvision 命名前缀
    model_save_path = os.path.join(checkpoint_dir, f"netvision_{safe_dataset_name}.pth")

    print("\n=== 开始训练 ===")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(args.device, dtype=torch.float32), labels.to(args.device, dtype=torch.long)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # -------------------
        # 验证阶段 (测试集)
        # -------------------
        model.eval()
        correct_test = 0
        total_test = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(args.device, dtype=torch.float32), labels.to(args.device, dtype=torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        val_loss = val_loss / len(test_loader)

        # 记录日志
        print(f"Epoch [{epoch + 1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {test_acc:.2f}%")

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', test_acc, epoch)

        # -------------------
        # 保存最佳模型权重
        # -------------------
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"    [!] 发现最佳模型 (Acc: {best_acc:.2f}%)，已保存至 {model_save_path}")

    print("=== 训练结束 ===")
    print(f"最高测试集准确率: {best_acc:.2f}%")
    writer.close()


if __name__ == "__main__":
    main()