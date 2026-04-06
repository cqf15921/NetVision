import os
import glob
import argparse
import numpy as np
from scapy.all import rdpcap, IP
from sklearn.model_selection import train_test_split


class LightGuardPreprocessor:
    def __init__(self, input_dir, output_idx_path, dataset_name, img_size=28, truncate_len=784):
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.truncate_len = truncate_len
        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

    def traffic_cleaning(self, packet):
        if IP in packet:
            return bytes(packet[IP])
        return None

    def traffic_truncation(self, raw_bytes):
        if len(raw_bytes) >= self.truncate_len:
            return raw_bytes[:self.truncate_len]
        else:
            return raw_bytes + b'\x00' * (self.truncate_len - len(raw_bytes))

    def pcap_to_images(self):
        images = []
        labels = []

        # 递归获取所有 pcap 和 pcapng 文件
        search_pattern_pcap = os.path.join(self.input_dir, '**', '*.pcap')
        search_pattern_pcapng = os.path.join(self.input_dir, '**', '*.pcapng')

        pcap_files = glob.glob(search_pattern_pcap, recursive=True) + \
                     glob.glob(search_pattern_pcapng, recursive=True)

        if not pcap_files:
            print(f"[!] 警告: 在 {self.input_dir} 下找不到任何 pcap/pcapng 文件。")
            return np.array([]), np.array([])

        for file_path in pcap_files:
            base_name = os.path.basename(file_path)
            parent_dir = os.path.basename(os.path.dirname(file_path))

            # 【智能标签提取逻辑】
            if self.dataset_name == 'ENTA_Datase':
                # ENTA 数据集：使用父文件夹名称作为标签 (Attacks / Normal Behavior)
                label_name = parent_dir
            else:
                # USTC 和 ToN-IoT：使用去掉后缀的文件名作为标签
                label_name = os.path.splitext(base_name)[0]

            print(f"正在处理: {base_name} (分配标签: {label_name})")

            try:
                # Scapy rdpcap 可以同时读取 .pcap 和 .pcapng 文件
                packets = rdpcap(file_path)
                sessions = packets.sessions()

                for session_name, session_pkts in sessions.items():
                    session_bytes = b''
                    for pkt in session_pkts:
                        cleaned_data = self.traffic_cleaning(pkt)
                        if cleaned_data:
                            session_bytes += cleaned_data

                    if len(session_bytes) == 0:
                        continue

                    truncated_data = self.traffic_truncation(session_bytes)
                    img_array = np.frombuffer(truncated_data, dtype=np.uint8).reshape(self.img_size, self.img_size)

                    images.append(img_array)
                    labels.append(label_name)

            except Exception as e:
                print(f"[!] 解析 {base_name} 出错: {e}")

        return np.array(images), np.array(labels)

    def save_as_idx(self, images, labels):
        if len(images) == 0:
            print("[!] 错误：没有提取到任何图像数据，无法保存。")
            return

        print(f"\n正在按 8:2 的比例划分 {self.dataset_name} 训练集和测试集...")

        # 过滤掉样本数少于 2 的极其稀有的会话类别（防止 train_test_split 报错）
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= 2]
        dropped_labels = unique_labels[counts < 2]

        if len(dropped_labels) > 0:
            print(f"[!] 自动过滤样本数不足 2 的极稀有类别: {list(dropped_labels)}")

        valid_indices = np.isin(labels, valid_labels)
        images = images[valid_indices]
        labels = labels[valid_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_path = self.output_idx_path.replace('.npz', '_train.npz')
        test_path = self.output_idx_path.replace('.npz', '_test.npz')

        np.savez_compressed(train_path, images=X_train, labels=y_train)
        np.savez_compressed(test_path, images=X_test, labels=y_test)

        print(f"[+] 训练集已保存至: {train_path} (样本数: {len(X_train)})")
        print(f"[+] 测试集已保存至: {test_path} (样本数: {len(X_test)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGuard PCAP Preprocessing")
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'ENTA_Datase', 'ToN-IoT', 'all'],
                        help='选择要处理的数据集名称')

    args = parser.parse_args()

    BASE_RAW_DIR = "data/raw"
    BASE_PROCESSED_DIR = "data/processed"
    os.makedirs(BASE_PROCESSED_DIR, exist_ok=True)

    # 动态构建需要处理的数据集列表
    datasets_to_process = ['USTC_TFC2016', 'ENTA_Datase', 'ToN-IoT'] if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets_to_process:
        print(f"\n{'=' * 40}")
        print(f"=== 开始执行 {ds_name} 流量预处理 ===")
        print(f"{'=' * 40}")
        raw_data_dir = os.path.join(BASE_RAW_DIR, ds_name)

        if not os.path.exists(raw_data_dir):
            print(f"[!] 错误: 找不到数据集目录 {raw_data_dir}")
            continue

        # 统一规范化保存文件名 (防止包含破折号等符号导致后续读取失败)
        safe_ds_name = ds_name.lower().replace('-', '_')
        output_path = os.path.join(BASE_PROCESSED_DIR, f"{safe_ds_name}_dataset.npz")

        preprocessor = LightGuardPreprocessor(raw_data_dir, output_path, dataset_name=ds_name)
        imgs, lbls = preprocessor.pcap_to_images()

        if len(imgs) > 0:
            preprocessor.save_as_idx(imgs, lbls)

    print("\n=== 所有请求的数据集均已预处理完成 ===")