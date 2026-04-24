import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import time
import os
import argparse
import numpy as np
from thop import profile  # 用于计算 FLOPs 和 参数量
import csv  # [新增] 用于导出 CSV 文件

# 引入可视化绘图库
import matplotlib.pyplot as plt
import seaborn as sns

# 导入你的模型和数据集加载类
from models.netvision_model import NetVision
from utils.dataset import NetVisionDataset


# ==========================================
# CustomNPZDataset 接收模型强制指定的 classes
# ==========================================
class CustomNPZDataset(Dataset):
    def __init__(self, npz_path, model_classes=None):
        data = np.load(npz_path, allow_pickle=True)
        self.images = data['images']
        self.labels_name = data['labels']

        # 优先使用模型自带的类别名单，保证维度和映射 100% 对齐
        if model_classes is not None:
            self.unique_labels = model_classes
        elif 'classes' in data:
            self.unique_labels = data['classes'].tolist()
        else:
            self.unique_labels = sorted(list(set(self.labels_name)))

        self.label_to_idx = {name: i for i, name in enumerate(self.unique_labels)}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # 增加通道维度 (H, W) -> (H, W, 1)
        image = image[:, :, np.newaxis]
        # 转换为 Tensor，并调整维度为 (C, H, W) 以匹配 PyTorch 卷积层需求
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0

        # 处理异常：如果测试集里出现了模型从未见过的流量类别，默认标记为 0 (或防崩溃机制)
        label = self.label_to_idx.get(self.labels_name[idx], 0)
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        return len(self.unique_labels)


def test():
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="NetVision 模型测试与评估脚本")

    # 核心参数：选择要评估的数据集
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        help='选择要评估的数据集')

    parser.add_argument('--model_type', type=str, default='netvision',
                        choices=['netvision', 'ghostnet', 'shufflenet', '1dcnn', 'noghost', 'resnet'],
                        help='选择要评估的模型架构')

    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # 新增：接收从 UI 传过来的自定义文件路径
    parser.add_argument('--custom_test_path', type=str, default=None, help='自定义测试集 (.npz) 路径')
    parser.add_argument('--custom_model_path', type=str, default=None, help='自定义模型权重 (.pth) 路径')

    args = parser.parse_args()

    # ==========================================
    # 2. 动态确定模型路径并提取类别名单
    # ==========================================
    if args.custom_test_path and args.custom_model_path:
        print(f"=== 开始执行自定义模型与数据评估 ===")
        model_path = args.custom_model_path
        eval_target_name = "自定义测试集"
    else:
        print(f"=== 开始在 {args.dataset} 上评估 NetVision ===")
        safe_dataset_name = args.dataset.lower().replace('-', '_')
        model_path = f'./checkpoints/{args.model_type}_{safe_dataset_name}.pth'
        eval_target_name = args.dataset

    if not os.path.exists(model_path):
        print(f"[!] 未找到模型权重文件: {model_path}")
        if not args.custom_model_path:
            print(f"[!] 请先执行: python train.py --dataset {args.dataset} 进行训练！")
        return

    # 加载权重，兼容只有 state_dict 的老版本和带有 classes 的新版本
    checkpoint = torch.load(model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'classes' in checkpoint:
        state_dict = checkpoint['state_dict']
        model_classes = checkpoint['classes']
    else:
        state_dict = checkpoint
        model_classes = None  # 如果加载的是老模型，无法获取固定的分类名

    # ==========================================
    # 3. 动态加载测试集 (注入模型指定的 classes)
    # ==========================================
    if args.custom_test_path and args.custom_model_path:
        try:
            test_dataset = CustomNPZDataset(args.custom_test_path, model_classes=model_classes)
        except Exception as e:
            print(f"[!] 加载自定义测试集失败: {e}")
            return
    else:
        processed_dir = "data/processed"
        try:
            test_dataset = NetVisionDataset(data_dir=processed_dir, dataset_name=args.dataset, is_train=False)
            # 内置模式下，强制对齐模型内存储的分类名单
            if model_classes is not None:
                test_dataset.unique_labels = model_classes
                test_dataset.label_to_idx = {name: i for i, name in enumerate(model_classes)}
                test_dataset.idx_to_label = {i: name for name, i in test_dataset.label_to_idx.items()}
        except FileNotFoundError as e:
            print(f"[!] 报错: {e}")
            print("[!] 未找到测试集数据，请先运行 utils/preprocessing.py 生成对应的 .npz 文件！")
            return

    num_classes = test_dataset.get_num_classes()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ==========================================
    # 4. 初始化并注入模型
    # ==========================================
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
        from models.noghost_model import NetVision as NoGhostNet
        model = NoGhostNet(num_classes=num_classes).to(args.device)
    elif args.model_type == 'resnet':
        from models.resnet_model import NetVision as ResNetModel
        model = ResNetModel(num_classes=num_classes).to(args.device)
    else:
        from models.netvision_model import NetVision
        model = NetVision(num_classes=num_classes).to(args.device)

    model.load_state_dict(state_dict)
    model.eval()

    # =====================================================================
    # 5. 计算模型的学术指标：FLOPs (计算复杂度) 和 Parameters (参数量)
    # =====================================================================
    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    flops_val, params_val = "N/A", "N/A"  # [新增] 为CSV保留变量
    try:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops_val = f"{flops / 1e6:.2f}"
        params_val = f"{params / 1e3:.2f}"
        print("\n" + "=" * 40)
        print("【模型复杂度评估】")
        print(f"计算复杂度 (FLOPs): {flops_val} M")
        print(f"模型参数量 (Params): {params_val} K")
        print("=" * 40 + "\n")
    except Exception as e:
        print(f"\n[!] 计算 FLOPs 失败: {e}\n")

    # =====================================================================
    # 6. 推理速度测试与性能评估
    # =====================================================================
    all_preds = []
    all_labels = []

    print(f"[*] 开始在 {eval_target_name} 上执行推理...")
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

    # 计算各项评估指标
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100

    print("-" * 40)
    print(f"【{eval_target_name} 推理效率】")
    print(f"测试样本总数: {len(test_dataset)}")
    print(f"推理总耗时:   {total_time:.2f} s")
    print(f"推理速度:     {flows_per_second:.2f} Flows/s")
    print("-" * 40)
    print(f"【{eval_target_name} 分类性能】")
    print(f"准确率 (Accuracy):  {accuracy:.2f}%")
    print(f"精确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall):    {recall:.2f}%")
    print(f"F1-measure:         {f1:.2f}%")
    print("-" * 40)

    target_names = [str(x) for x in test_dataset.unique_labels]
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # =====================================================================
    # 7. 生成各项指标的可视化图表、专属文件夹及保存 CSV [主要修改部分]
    # =====================================================================
    print("[*] 正在生成评估结果和可视化图表...")

    # 基础 UI 目录
    os.makedirs('results', exist_ok=True)

    # [新增] 按照 数据集+模型名称 创建专属输出文件夹
    specific_dir_name = f"{eval_target_name}_{args.model_type}".replace(' ', '_')
    specific_res_dir = os.path.join('results', specific_dir_name)
    os.makedirs(specific_res_dir, exist_ok=True)

    # (1) 性能指标柱状图
    plt.figure(figsize=(8, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [accuracy, precision, recall, f1]

    sns.barplot(x=metrics_names, y=metrics_values, palette="viridis")
    plt.title(f'{eval_target_name} ({args.model_type}) Performance Metrics (%)')
    plt.ylim(0, 100)

    for i, v in enumerate(metrics_values):
        plt.text(i, v + 1.5, f"{v:.2f}%", color='black', ha='center', fontweight='bold')

    plt.tight_layout()
    # 为保证 UI 正常显示保留旧版输出
    plt.savefig('results/metrics_bar.png', dpi=300)
    # [新增] 保存到专属文件夹
    plt.savefig(os.path.join(specific_res_dir, 'metrics_bar.png'), dpi=300)
    plt.close()

    # (2) 混淆矩阵热力图
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix on {eval_target_name} ({args.model_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.savefig(os.path.join(specific_res_dir, 'confusion_matrix.png'), dpi=300)  # [新增] 保存到专属文件夹
    plt.close()

    # (3) [新增] 导出评估指标至 CSV 文件
    csv_metrics_path = os.path.join(specific_res_dir, 'overall_metrics.csv')
    with open(csv_metrics_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Model', 'Accuracy(%)', 'Precision(%)', 'Recall(%)', 'F1-Score(%)',
                         'FLOPs(M)', 'Params(K)', 'Inference_Time(s)', 'Flows_per_second'])
        writer.writerow([eval_target_name, args.model_type, f"{accuracy:.2f}", f"{precision:.2f}",
                         f"{recall:.2f}", f"{f1:.2f}", flops_val, params_val, f"{total_time:.2f}",
                         f"{flows_per_second:.2f}"])

    # (4) [新增] 导出详细分类报告至 CSV 文件
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0,
                                        output_dict=True)
    csv_report_path = os.path.join(specific_res_dir, 'classification_report.csv')
    with open(csv_report_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for cls_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                writer.writerow([cls_name, f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}",
                                 f"{metrics['f1-score']:.4f}", metrics['support']])
            else:
                writer.writerow([cls_name, "", "", f"{metrics:.4f}", ""])  # 兼容 accuracy 单值

    print(f"[*] 图表已存至专属目录: {specific_res_dir}")
    print(f"[*] 各项评估指标和详细分类报告已成功导出为 CSV: {specific_res_dir} 目录！")


if __name__ == "__main__":
    test()