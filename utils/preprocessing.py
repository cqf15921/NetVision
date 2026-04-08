import os
import glob
import argparse
import numpy as np
from collections import defaultdict
# 加入了 IPv6 支持，防止 IoT 数据集中的正常设备流量被漏掉
from scapy.all import IP, IPv6, TCP, UDP
from scapy.utils import PcapReader
from sklearn.model_selection import train_test_split


class NetVisionPreprocessor:
    def __init__(self, input_dir, output_idx_path, dataset_name, img_size=28, truncate_len=784):
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.truncate_len = truncate_len
        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

    def traffic_cleaning(self, packet):
        # 同时支持 IPv4 和 IPv6 层的提取
        if packet.haslayer(IP):
            return bytes(packet[IP])
        elif packet.haslayer(IPv6):
            return bytes(packet[IPv6])
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

            # 【智能标签提取逻辑】
            # 因为 CIC_IoT_2023, USTC 等数据集 pcap 文件直接在文件夹下，所以直接使用去掉后缀的文件名作为标签
            label_name = os.path.splitext(base_name)[0]

            print(f"\n正在处理: {base_name} (分配标签: {label_name})")

            # === 核心优化 1：内存优化的流式读取引擎 ===
            sessions = defaultdict(bytes)

            # === 核心优化 2：数据重采样与防过拟合截断 ===
            max_packets_per_file = 300000  # 每个文件最多只读取前 30 万个数据包 (可根据需要修改)

            try:
                with PcapReader(file_path) as pcap_reader:
                    i = 0
                    while True:
                        # === 核心优化 3：异常与畸形包跳过机制 (抗损毁) ===
                        try:
                            pkt = pcap_reader.read_packet()
                        except EOFError:
                            break
                        except Exception:
                            continue

                        if i >= max_packets_per_file:
                            print(f"    [!] 达到采样上限 ({max_packets_per_file}包)，触发防过拟合截断，停止读取。")
                            break

                        if i > 0 and i % 10000 == 0:
                            print(f"    ... 已读取 {i} 个数据包 ...")

                        cleaned_data = self.traffic_cleaning(pkt)
                        if not cleaned_data:
                            i += 1
                            continue

                        # === 支持五元组的 IPv4 和 IPv6 混合提取 ===
                        has_ip4 = pkt.haslayer(IP)
                        has_ip6 = pkt.haslayer(IPv6)

                        if has_ip4 or has_ip6:
                            net_layer = pkt[IP] if has_ip4 else pkt[IPv6]

                            src = net_layer.src
                            dst = net_layer.dst
                            proto = net_layer.proto if has_ip4 else net_layer.nh
                            sport, dport = 0, 0

                            if pkt.haslayer(TCP):
                                sport = pkt[TCP].sport
                                dport = pkt[TCP].dport
                            elif pkt.haslayer(UDP):
                                sport = pkt[UDP].sport
                                dport = pkt[UDP].dport

                            end1 = f"{src}:{sport}"
                            end2 = f"{dst}:{dport}"
                            session_key = f"{proto}-" + "-".join(sorted([end1, end2]))

                            if len(sessions[session_key]) < self.truncate_len:
                                sessions[session_key] += cleaned_data

                        i += 1

                print(f"    [+] {base_name} 读取完毕！共提取 {len(sessions)} 个有效会话，正在转换为图像矩阵...")
                for sess_key, session_bytes in sessions.items():
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
    parser = argparse.ArgumentParser(description="NetVision PCAP Preprocessing (Streaming & Optimized)")
    # 将 ENTA_Datase 替换为 CIC_IoT_2023
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        choices=['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT', 'all'],
                        help='选择要处理的数据集名称')

    args = parser.parse_args()

    BASE_RAW_DIR = "data/raw"
    BASE_PROCESSED_DIR = "data/processed"
    os.makedirs(BASE_PROCESSED_DIR, exist_ok=True)

    datasets_to_process = ['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT'] if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets_to_process:
        print(f"\n{'=' * 40}")
        print(f"=== 开始执行 {ds_name} 流量预处理 ===")
        print(f"{'=' * 40}")
        raw_data_dir = os.path.join(BASE_RAW_DIR, ds_name)

        if not os.path.exists(raw_data_dir):
            print(f"[!] 错误: 找不到数据集目录 {raw_data_dir}")
            continue

        safe_ds_name = ds_name.lower().replace('-', '_')
        output_path = os.path.join(BASE_PROCESSED_DIR, f"{safe_ds_name}_dataset.npz")

        preprocessor = NetVisionPreprocessor(raw_data_dir, output_path, dataset_name=ds_name)
        imgs, lbls = preprocessor.pcap_to_images()

        if len(imgs) > 0:
            preprocessor.save_as_idx(imgs, lbls)

    print("\n=== 所有请求的数据集均已预处理完成 ===")