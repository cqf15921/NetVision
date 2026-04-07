import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import time
import os
import argparse
from thop import profile  # 用于计算 FLOPs 和 参数量

# 导入你的模型和数据集加载类 (已更新为 NetVision)
from models.netvision_model import NetVision
from utils.dataset import NetVisionDataset


def test():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="NetVision 模型测试与评估脚本")

    # 核心参数：选择要评估的数据集 (已更新为最新的 PCAP 数据集列表)
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'ENTA_Datase', 'ToN-IoT'],
                        help='选择要评估的数据集')

    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    print(f"=== 开始在 {args.dataset} 上评估 NetVision ===")

    # ==========================================
    # 2. 加载测试集
    # ==========================================
    processed_dir = "data/processed"
    try:
        test_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=False)
    except FileNotFoundError as e:
        print(f"[!] 报错: {e}")
        print("[!] 未找到测试集数据，请先运行 utils/preprocessing.py 生成对应的 .npz 文件！")
        return

    num_classes = test_dataset.get_num_classes()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 3. 动态加载模型与权重
    # ==========================================
    # 统一规范化名字，与 train.py 的保存逻辑保持一致
    safe_dataset_name = args.dataset.lower().replace('-', '_')

    # 读取的权重文件名前缀更新为 netvision
    model_path = f'./checkpoints/netvision_{safe_dataset_name}.pth'

    model = NetVision().to(args.device)

    # 动态适配输出层类别数
    model.f1[2] = nn.Linear(256, num_classes).to(args.device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"[*] 成功加载模型权重: {model_path}")
    else:
        print(f"[!] 未找到模型权重文件: {model_path}")
        print(f"[!] 请先执行: python train.py --dataset {args.dataset} 进行训练！")
        return

    model.eval()

    # =====================================================================
    # 4. 计算模型的学术指标：FLOPs (计算复杂度) 和 Parameters (参数量)
    # =====================================================================
    # 构造假输入：(BatchSize, Channels, Height, Width) -> 对应 784 字节的 28x28 灰度图
    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    print("\n" + "=" * 40)
    print("【模型复杂度评估】")
    print(f"计算复杂度 (FLOPs): {flops / 1e6:.2f} M")
    print(f"模型参数量 (Params): {params / 1e3:.2f} K")
    print("=" * 40 + "\n")

    # =====================================================================
    # 5. 推理速度测试与性能评估
    # =====================================================================
    all_preds = []
    all_labels = []

    print(f"[*] 开始在 {args.dataset} 测试集上执行推理...")
    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(args.device, dtype=torch.float32)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    end_time = time.time()
    total_time = end_time - start_time
    flows_per_second = len(test_dataset) / total_time if total_time > 0 else 0

    # 6. 计算各项评估指标并打印报告
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    print("-" * 40)
    print(f"【{args.dataset} 推理效率】")
    print(f"测试样本总数: {len(test_dataset)}")
    print(f"推理总耗时:   {total_time:.2f} s")
    print(f"推理速度:     {flows_per_second:.2f} Flows/s")
    print("-" * 40)
    print(f"【{args.dataset} 分类性能】")
    print(f"准确率 (Accuracy):  {accuracy:.2f}%")
    print(f"精确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall):    {recall:.2f}%")
    print(f"F1-measure:         {f1:.2f}%")
    print("-" * 40)

    # 打印详细分类报告 (防止 numpy 标签报错，统一转换为字符串)
    target_names = [str(x) for x in test_dataset.unique_labels]
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))


if __name__ == "__main__":
    test()