import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import time
import os
import numpy as np
from thop import profile  # 新增：用于计算 FLOPs 和 参数量

from models.lightguard_model import LightGuard
from utils.dataset import TrafficDataset


def test():
    # 1. 环境配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './checkpoints/lightguard_ustc.pth'
    batch_size = 64

    # 2. 加载数据集
    # 注意：这里严格加载由 preprocessing.py 划分出来的独立测试集 _test.npz
    test_data_path = "data/processed/ustc_tfc2016_dataset_test.npz"
    if not os.path.exists(test_data_path):
        print(f"找不到测试集文件 {test_data_path}，请确认是否已重新运行 preprocessing.py")
        return

    dataset = TrafficDataset(test_data_path)
    num_classes = dataset.get_num_classes()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 3. 加载模型并调整输出层
    model = LightGuard().to(device)
    model.f1[2] = nn.Linear(256, num_classes).to(device)  # 适配动态类别数 (如 USTC 的 20 类)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    else:
        print("未找到模型文件，请先运行 train.py 进行训练")
        return

    model.eval()

    # =====================================================================
    # 4. 计算模型的学术指标：FLOPs (计算复杂度) 和 Parameters (参数量)
    # =====================================================================
    # 构造一个形状为 (BatchSize, Channels, Height, Width) 的假输入
    # 对应论文中 784 字节的截断，即 28x28 的单通道灰度图
    dummy_input = torch.randn(1, 1, 28, 28).to(device)

    # verbose=False 用于关闭 thop 默认的逐层打印输出，保持控制台整洁
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    print("\n" + "=" * 40)
    print("【模型复杂度评估】")
    print(f"计算复杂度 (FLOPs): {flops / 1e6:.2f} M")  # 论文中 LightGuard 约为 1.25 M
    print(f"模型参数量 (Params): {params / 1e3:.2f} K")  # 论文中 LightGuard 约为 78 K
    print("=" * 40 + "\n")

    # =====================================================================
    # 5. 推理速度测试 (Flows/s) 及 性能评估指标收集
    # =====================================================================
    all_preds = []
    all_labels = []

    print("开始在测试集上评估推理速度与准确率...")
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
    # Flows/s：每秒处理的流(样本)数量
    flows_per_second = len(dataset) / total_time

    # =====================================================================
    # 6. 计算各项评估指标并打印报告
    # =====================================================================
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    print("-" * 40)
    print("【推理效率表现】")
    print(f"测试样本总数: {len(dataset)}")
    print(f"推理总耗时:   {total_time:.2f} s")
    print(f"推理速度:     {flows_per_second:.2f} Flows/s")
    print("-" * 40)
    print("【分类性能表现】")
    print(f"准确率 (Accuracy):  {accuracy:.2f}%")  # 论文 USTC 结果期望为 99.94% 左右
    print(f"精确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall):    {recall:.2f}%")
    print(f"F1-measure:         {f1:.2f}%")
    print("-" * 40)

    # 打印详细分类报告 (包含每个单独类别的表现)
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=dataset.unique_labels, zero_division=0))


if __name__ == "__main__":
    test()