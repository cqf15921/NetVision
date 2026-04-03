import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class LightGuardDataset(Dataset):
    def __init__(self, data_dir, dataset_name="USTC_TFC2016", is_train=True, transform=None):
        """
        :param data_dir: 预处理数据的根目录 (如 'data/processed')
        :param dataset_name: 数据集名称
        :param is_train: True 加载训练集，False 加载测试集
        :param transform: PyTorch 图像变换
        """
        # 动态拼接文件名 (小写化并将连字符替换为下划线，以匹配 preprocessing.py 的输出)
        # 这一步修复了加载 ToN-IoT 数据集时文件名不匹配的问题
        mode = "train" if is_train else "test"
        npz_filename = f"{dataset_name.lower().replace('-', '_')}_dataset_{mode}.npz"
        npz_path = os.path.join(data_dir, npz_filename)

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[!] 找不到数据集文件: {npz_path}。请确认是否已运行 preprocessing.py")

        # 加载预处理后的数据
        data = np.load(npz_path, allow_pickle=True)
        self.images = data['images']
        self.labels_name = data['labels']

        # 动态获取所有唯一的类别名称并排序，建立名称到索引的映射
        self.unique_labels = sorted(list(set(self.labels_name)))
        self.label_to_idx = {name: i for i, name in enumerate(self.unique_labels)}
        self.idx_to_label = {i: name for name, i in self.label_to_idx.items()}

        # 默认变换：转换为张量并归一化
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label_name = self.labels_name[idx]

        if self.transform:
            # 增加通道维度 (H, W) -> (H, W, 1) 以适配 ToTensor
            image = image[:, :, np.newaxis]
            image = self.transform(image)

        label = self.label_to_idx[label_name]
        return image, torch.tensor(label, dtype=torch.long)

    def get_num_classes(self):
        """返回当前数据集的类别总数"""
        return len(self.unique_labels)