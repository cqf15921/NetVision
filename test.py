import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
import time
import numpy as np
from models.lightguard_model import LightGuard
from utils.dataset import TrafficDataset


def test():
    # 1. 环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './checkpoints/lightguard_ustc.pth'
    batch_size = 64

    # 2. 加载数据集
    dataset = TrafficDataset("data/processed/ustc_tfc2016_dataset.npz")
    num_classes = dataset.get_num_classes()

    # 论文中实验重复了10次取平均值，这里展示单次测试逻辑
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. 加载模型并调整输出层
    model = LightGuard().to(device)
    model.f1[2] = nn.Linear(256, num_classes).to(device)  # 适配 20 类

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print("未找到模型文件，请先运行 train.py")
        return

    model.eval()

    all_preds = []
    all_labels = []

    # 4. 推理速度测试 (Flows/s) [cite: 445, 551]
    print("开始评估推理速度...")
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    end_time = time.time()
    total_time = end_time - start_time
    flows_per_second = len(dataset) / total_time

    # 5. 计算各项指标 [cite: 423, 426, 427, 428, 431]
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro') * 100
    recall = recall_score(all_labels, all_preds, average='macro') * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print("-" * 30)
    print(f"测试样本总数: {len(dataset)}")
    print(f"推理总耗时: {total_time:.2f}s")
    print(f"推理速度 (Flows/s): {flows_per_second:.2f}")  # 论文中 LightGuard 约为 440 Flows/s
    print("-" * 30)
    print(f"准确率 (Accuracy): {accuracy:.2f}%")  # 论文 USTC 结果为 99.94%
    print(f"精确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall): {recall:.2f}%")
    print(f"F1-measure: {f1:.2f}%")
    print("-" * 30)

    # 打印详细分类报告 (包含每个类别的表现)
    print("详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=dataset.unique_labels))


if __name__ == "__main__":
    import os

    test()