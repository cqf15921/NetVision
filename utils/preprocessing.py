import os
import numpy as np
from scapy.all import rdpcap, IP
from sklearn.model_selection import train_test_split


class LightGuardPreprocessor:
    def __init__(self, input_dir, output_idx_path, img_size=28, truncate_len=784):
        """
        :param input_dir: 原始 pcap 文件夹路径 (例如 data/raw/USTC_TFC2016)
        :param output_idx_path: 转换后的 NPZ 文件基础保存路径
        :param img_size: 生成图像的尺寸 (28x28)
        :param truncate_len: 截断长度 (784字节，28*28=784)
        """
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.img_size = img_size
        self.truncate_len = truncate_len

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

    def traffic_cleaning(self, packet):
        """
        流量清洗阶段 [cite: 199]：
        移除 MAC 地址等可能影响分类的链路层信息。
        通过提取 IP 层及以上的数据来实现清洗。
        """
        if IP in packet:
            # 仅保留网络层(IP)及以上的数据，有效去除数据链路层(MAC)
            return bytes(packet[IP])
        return None

    def traffic_truncation(self, raw_bytes):
        """
        流量截断阶段 [cite: 200, 204]：
        将拼接好的会话流量处理为固定的 m 字节（此处为 784 字节）。
        若超过则截断，不足则在末尾补零（\x00）。
        """
        if len(raw_bytes) >= self.truncate_len:
            return raw_bytes[:self.truncate_len]
        else:
            return raw_bytes + b'\x00' * (self.truncate_len - len(raw_bytes))

    def pcap_to_images(self):
        """
        执行完整预处理：切割(Session)、清洗、截断并转换为灰度图 [cite: 147, 197]。
        """
        images = []
        labels = []

        # 遍历数据集结构：Benign/ 和 Malware/
        for category in ['Benign', 'Malware']:
            cat_path = os.path.join(self.input_dir, category)
            if not os.path.exists(cat_path):
                print(f"警告: 找不到目录 {cat_path}")
                continue

            pcap_files = [f for f in os.listdir(cat_path) if f.endswith('.pcap')]

            for pcap_file in pcap_files:
                print(f"正在处理: {category}/{pcap_file}")
                file_path = os.path.join(cat_path, pcap_file)

                # 标签：根据文件名定义，例如 "BitTorrent"
                label_name = pcap_file.split('.')[0]

                try:
                    # 读取 pcap 文件
                    packets = rdpcap(file_path)

                    # 1. 流量切割 (Traffic Segmentation)：按五元组将包划分为会话(Session)
                    sessions = packets.sessions()

                    for session_name, session_pkts in sessions.items():
                        session_bytes = b''

                        for pkt in session_pkts:
                            # 2. 流量清洗 (Traffic Cleaning)：提取有效载荷 [cite: 199]
                            cleaned_data = self.traffic_cleaning(pkt)
                            if cleaned_data:
                                session_bytes += cleaned_data

                        # 如果该会话提取不到有效IP数据，跳过
                        if len(session_bytes) == 0:
                            continue

                        # 3. 流量截断 (Traffic Truncation)：对整个会话截断为 784 字节 [cite: 204]
                        truncated_data = self.traffic_truncation(session_bytes)

                        # 4. 数据集生成 (Dataset Generation)：转换为 28x28 灰度图 [cite: 207]
                        img_array = np.frombuffer(truncated_data, dtype=np.uint8).reshape(self.img_size, self.img_size)

                        images.append(img_array)
                        labels.append(label_name)

                except Exception as e:
                    print(f"解析 {pcap_file} 出错: {e}")

        return np.array(images), np.array(labels)

    def save_as_idx(self, images, labels):
        """
        数据集划分与保存：
        按照 8:2 划分训练集和测试集，分别保存为独立的 npz 文件，以防止数据泄露。
        """
        if len(images) == 0:
            print("错误：没有提取到任何图像数据，无法保存。")
            return

        print("正在按 8:2 的比例划分训练集和测试集...")
        # 使用 stratify 保证切分后各类别比例一致
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # 构造训练集和测试集的保存路径
        train_path = self.output_idx_path.replace('.npz', '_train.npz')
        test_path = self.output_idx_path.replace('.npz', '_test.npz')

        # 分别保存
        np.savez(train_path, images=X_train, labels=y_train)
        np.savez(test_path, images=X_test, labels=y_test)

        print(f"训练集已保存至: {train_path} (样本数: {len(X_train)})")
        print(f"测试集已保存至: {test_path} (样本数: {len(X_test)})")


if __name__ == "__main__":
    # 使用示例
    # 假设你在项目根目录运行此脚本
    raw_data_dir = "data/raw/USTC_TFC2016"
    output_path = "data/processed/ustc_tfc2016_dataset.npz"

    print("=== 开始执行 LightGuard 流量预处理 ===")
    preprocessor = LightGuardPreprocessor(raw_data_dir, output_path)

    # 执行流量转换为图像
    imgs, lbls = preprocessor.pcap_to_images()

    # 划分并保存为 _train.npz 和 _test.npz
    preprocessor.save_as_idx(imgs, lbls)
    print("=== 流量预处理完成 ===")