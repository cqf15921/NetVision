import os
import binascii
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
import struct


class LightGuardPreprocessor:
    def __init__(self, input_dir, output_idx_path, img_size=28, truncate_len=784):
        """
        :param input_dir: 原始 pcap 文件夹路径 (例如 data/raw/ustc_tfc2016)
        :param output_idx_path: 转换后的 IDX 文件保存路径
        :param img_size: 生成图像的尺寸 (28x28)
        :param truncate_len: 截断长度 (784字节)
        """
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.img_size = img_size
        self.truncate_len = truncate_len

    def traffic_cleaning(self, packet):
        """
        流量清洗阶段 [cite: 199]：
        移除 MAC 地址等可能影响分类的链路层信息 。
        这里通过提取 IP 层及以上的数据来实现清洗。
        """
        if IP in packet:
            # 仅保留网络层及以上的数据，有效去除 MAC 地址
            return bytes(packet[IP])
        return None

    def traffic_truncation(self, raw_bytes):
        """
        流量截断阶段 [cite: 200]：
        将流量处理为固定的 m 字节（此处为 784 字节） [cite: 204]。
        若超过则截断，不足则补零 。
        """
        if len(raw_bytes) >= self.truncate_len:
            return raw_bytes[:self.truncate_len]
        else:
            return raw_bytes + b'\x00' * (self.truncate_len - len(raw_bytes))

    def pcap_to_images(self):
        """
        执行完整预处理：切割、清洗、截断并转换为灰度图 [cite: 147, 197]。
        """
        images = []
        labels = []

        # 遍历数据集结构：Benign/ 和 Malware/
        for category in ['Benign', 'Malware']:
            cat_path = os.path.join(self.input_dir, category)
            if not os.path.exists(cat_path): continue

            pcap_files = [f for f in os.listdir(cat_path) if f.endswith('.pcap')]

            for pcap_file in pcap_files:
                print(f"正在处理: {pcap_file}")
                file_path = os.path.join(cat_path, pcap_file)

                # 标签：根据文件名或文件夹定义（此处简化处理）
                # 论文中提到需要手动标注类别
                label_name = pcap_file.split('.')[0]

                try:
                    packets = rdpcap(file_path)
                    for pkt in packets:
                        # 1. 清洗 [cite: 199]
                        cleaned_data = self.traffic_cleaning(pkt)
                        if not cleaned_data: continue

                        # 2. 截断 [cite: 200]
                        truncated_data = self.traffic_truncation(cleaned_data)

                        # 3. 灰度图生成 [cite: 197]
                        # 将 784 字节转换为 28x28 的矩阵
                        img_array = np.frombuffer(truncated_data, dtype=np.uint8).reshape(self.img_size, self.img_size)
                        images.append(img_array)
                        labels.append(label_name)  # 实际复现时建议将其映射为数字 ID
                except Exception as e:
                    print(f"解析 {pcap_file} 出错: {e}")

        return np.array(images), labels

    def save_as_idx(self, images, labels):
        """
        数据集生成阶段 [cite: 196]：
        将图像和标签转换为 IDX 格式以便高效加载 [cite: 208]。
        """
        # 注意：此处为简化逻辑，实际 IDX 存储需遵循特定二进制协议
        # 建议直接使用 numpy 或 pickle 保存，或实现标准的 MNIST IDX 格式
        np.savez(self.output_idx_path, images=images, labels=labels)
        print(f"数据集已保存至: {self.output_idx_path}")


if __name__ == "__main__":
    # 使用示例
    raw_data_dir = "data/raw/ustc_tfc2016"
    output_path = "data/processed/ustc_tfc2016_dataset.npz"

    preprocessor = LightGuardPreprocessor(raw_data_dir, output_path)
    imgs, lbls = preprocessor.pcap_to_images()
    preprocessor.save_as_idx(imgs, lbls)