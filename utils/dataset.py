import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class TrafficDataset(Dataset):
    def __init__(self, npz_path, transform=None):
        """
        :param npz_path: 预处理生成的 .npz 文件路径
        :param transform: PyTorch 图像变换 (例如 ToTensor)
        """
        # 加载预处理后的数据 [cite: 208]
        data = np.load(npz_path, allow_pickle=True)
        self.images = data['images']
        self.labels_name = data['labels']

        # 获取所有唯一的类别名称并排序，建立名称到索引的映射
        self.unique_labels = sorted(list(set(self.labels_name)))
        self.label_to_idx = {name: i for i, name in enumerate(self.unique_labels)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

        # 默认变换：转换为张量并归一化到 [0, 1]
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_name = self.labels_name[idx]

        # 将图像转换为 PIL Image 格式以便 transform 处理
        # 图像已截断为 784 字节并重塑为 28x28 [cite: 392, 401]
        if self.transform:
            # 增加通道维度 (H, W) -> (H, W, 1)
            image = image[:, :, np.newaxis]
            image = self.transform(image)

        label = self.label_to_idx[label_name]
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        """返回类别总数，用于修改模型输出层"""
        return len(self.unique_labels)


# 打印 USTC-TFC2016 的参考类别映射
def get_ustc_mapping():
    return [
        "BitTorrent", "FaceTime", "FTP", "Gmail", "Mysql",
        "Outlook", "Skype", "SMB", "Weibo", "World",  # 良性
        "Cridex", "Geodo", "Htbot", "Miuref", "Neris",
        "Nsis", "Shifu", "Tinba", "Virut", "Zeus"  # 恶意
    ]


if __name__ == "__main__":
    # 测试代码
    try:
        dataset = TrafficDataset("data/processed/ustc_tfc2016_dataset.npz")
        print(f"成功加载数据集，样本数: {len(dataset)}")
        print(f"类别总数: {dataset.get_num_classes()}")
        print(f"类别映射示例: {dataset.label_to_idx}")

        img, label = dataset[0]
        print(f"单张图像形状: {img.shape}, 标签索引: {label}")
    except Exception as e:
        print(f"测试失败（请先运行 preprocessing.py）: {e}")