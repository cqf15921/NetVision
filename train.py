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

    # 核心参数
    # 【优化点 1】：删除了 choices 限制，允许接收来自 UI 界面的 "User_Dataset"
    parser.add_argument('--dataset', type=str, default='CIC_IoT_2023',
                        help='选择要训练的数据集名称')
    parser.add_argument('--model_type', type=str, default='netvision',
                        choices=['netvision', 'ghostnet', 'shufflenet', '1dcnn', 'noghost', 'resnet'],
                        help='选择要训练的模型架构')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='训练设备')

    # 控制打印频率的参数
    parser.add_argument('--log_interval', type=int, default=1000, help='每隔多少个 step 打印一次日志')

    args = parser.parse_args()

    # 1. 动态确定路径与统一命名格式
    safe_dataset_name = args.dataset.lower().replace('-', '_')
    processed_dir = "data/processed"

    # 2. 加载数据集
    print(f"[*] 正在加载数据集: {args.dataset}...")
    try:
        train_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        test_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"[!] 错误: 无法加载数据，请确保已运行预处理脚本。{e}")
        return

    num_classes = train_dataset.get_num_classes()
    print(f"[*] 类别数: {num_classes} | 训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    # 【优化点 2】：拦截单类别训练，防止 CrossEntropyLoss 发生 Target out of bounds 崩溃
    if num_classes < 2:
        print("\n[!] 🚨 致命错误：当前数据集的有效类别数不足 2 个。")
        print(
            "[!] 深度学习分类模型至少需要 2 种不同的流量（如 1种正常流量 + 1种攻击流量）才能进行训练。请上传更多种类的流量包！")
        return

    # 3. 动态初始化模型
    print(f"[*] 正在加载 {args.model_type.upper()} 模型架构...")
    if args.model_type == 'ghostnet':
        from models.ghostnet_model import GhostNet
        model = GhostNet(num_classes=num_classes).to(args.device)
    elif args.model_type == 'shufflenet':
        from models.shufflenet_model import ShuffleNetV2
        model = ShuffleNetV2(num_classes=num_classes).to(args.device)
    elif args.model_type == '1dcnn':
        from models.cnn1d_model import CNN1D
        model = CNN1D(num_classes=num_classes).to(args.device)
    elif args.model_type == 'noghost':
        # 从 noghost_model 导入并起别名
        from models.noghost_model import NetVision as NoGhostNet
        model = NoGhostNet(num_classes=num_classes).to(args.device)
    elif args.model_type == 'resnet':
        # 从 resnet_model 导入并起别名
        from models.resnet_model import NetVision as ResNetModel
        model = ResNetModel(num_classes=num_classes).to(args.device)
    else:
        # 默认原版 NetVision
        from models.netvision_model import NetVision
        model = NetVision(num_classes=num_classes).to(args.device)

    # 4. 优化策略
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print(f"[*] 开始训练 (频率: 每 {args.log_interval} steps 打印一次)...")
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

            if (i + 1) % args.log_interval == 0:
                print(f'Epoch [{epoch + 1:2d}/{args.epochs}] | Step [{i + 1:5d}/{len(train_loader)}] | '
                      f'Loss: {loss.item():.4f} | Train Acc: {100. * correct_train / total_train:.2f}%')

        # 更新学习率
        scheduler.step()
        train_acc = 100. * correct_train / total_train
        train_loss_avg = running_loss / len(train_loader)

        # ====================
        # [B] 验证阶段
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

        # 每个 Epoch 结束时的固定总结打印
        print(f'==> Epoch {epoch + 1} 结束 | '
              f'Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')

        # ====================
        # [C] 保存最佳模型
        # ====================
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = f'checkpoints/{args.model_type}_{safe_dataset_name}.pth'

            # 【优化点 3】：将权重和当前数据集的全局类别名单一起打包保存
            checkpoint = {
                'state_dict': model.state_dict(),
                'classes': train_dataset.unique_labels
            }
            torch.save(checkpoint, save_path)
            print(f'    [+] 发现最佳模型 (Val Acc: {best_acc:.2f}%)，权重与类别名单已保存。')

    print(f"\n[*] 训练任务完成! 最终最佳测试集准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    train()