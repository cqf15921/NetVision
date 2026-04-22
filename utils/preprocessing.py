import os
import glob
import argparse
import shutil  # 用于清理临时目录
import numpy as np
from collections import defaultdict
from scapy.all import IP, IPv6, TCP, UDP
from scapy.utils import PcapReader
from sklearn.model_selection import train_test_split


class NetVisionPreprocessor:
    def __init__(self, input_dir, output_idx_path, dataset_name, max_packets=0, img_size=28, truncate_len=784):
        self.input_dir = input_dir
        self.output_idx_path = output_idx_path
        self.dataset_name = dataset_name
        self.max_packets = max_packets
        self.img_size = img_size
        self.truncate_len = truncate_len

        os.makedirs(os.path.dirname(self.output_idx_path), exist_ok=True)

        # 临时缓存目录
        self.temp_dir = os.path.join(os.path.dirname(self.output_idx_path), f"temp_{self.dataset_name}")
        os.makedirs(self.temp_dir, exist_ok=True)

    def traffic_cleaning(self, packet):
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

    def process_all_pcaps(self):
        """逐个文件提取并缓存，采用原子化写入策略"""
        search_pattern_pcap = os.path.join(self.input_dir, '**', '*.pcap')
        search_pattern_pcapng = os.path.join(self.input_dir, '**', '*.pcapng')

        pcap_files = glob.glob(search_pattern_pcap, recursive=True) + \
                     glob.glob(search_pattern_pcapng, recursive=True)

        if not pcap_files:
            print(f"[!] 警告: 在 {self.input_dir} 下找不到任何 pcap/pcapng 文件。")
            return False

        for file_path in pcap_files:
            base_name = os.path.basename(file_path)

            # USTC_TFC2016 的父目录只有 Malware 和 Benign 两个，因此必须继续使用文件名作为标签
            if self.dataset_name == 'USTC_TFC2016':
                label_name = os.path.splitext(base_name)[0]
            else:
                # 对于 CIC_IoT_2023 和 ToN-IoT，使用文件所在的上一级目录名作为真实标签
                label_name = os.path.basename(os.path.dirname(file_path))

            # 最终路径
            temp_file_path = os.path.join(self.temp_dir, f"{label_name}_{base_name}.npz")

            # 断点续传：只有正式的 .npz 存在才跳过
            if os.path.exists(temp_file_path):
                print(f"\n[跳过] 缓存已存在: {base_name}")
                continue

            print(f"\n正在处理: {base_name} (标签: {label_name})")

            sessions = defaultdict(bytes)
            images = []
            labels = []

            try:
                with PcapReader(file_path) as pcap_reader:
                    i = 0
                    while True:
                        try:
                            pkt = pcap_reader.read_packet()
                        except EOFError:
                            break
                        except Exception:
                            continue

                        if self.max_packets > 0 and i >= self.max_packets:
                            print(f"    [!] 达到采样上限，停止读取。")
                            break

                        cleaned_data = self.traffic_cleaning(pkt)
                        if not cleaned_data:
                            i += 1
                            continue

                        has_ip4 = pkt.haslayer(IP)
                        has_ip6 = pkt.haslayer(IPv6)

                        if has_ip4 or has_ip6:
                            net_layer = pkt[IP] if has_ip4 else pkt[IPv6]
                            src, dst = net_layer.src, net_layer.dst
                            proto = net_layer.proto if has_ip4 else net_layer.nh
                            sport, dport = 0, 0
                            if pkt.haslayer(TCP):
                                sport, dport = pkt[TCP].sport, pkt[TCP].dport
                            elif pkt.haslayer(UDP):
                                sport, dport = pkt[UDP].sport, pkt[UDP].dport

                            session_key = f"{proto}-" + "-".join(sorted([f"{src}:{sport}", f"{dst}:{dport}"]))
                            if len(sessions[session_key]) < self.truncate_len:
                                sessions[session_key] += cleaned_data
                        i += 1

                for sess_key, session_bytes in sessions.items():
                    if len(session_bytes) == 0: continue
                    img_array = np.frombuffer(self.traffic_truncation(session_bytes), dtype=np.uint8).reshape(
                        self.img_size, self.img_size)
                    images.append(img_array)
                    labels.append(label_name)

                # --- 改进：原子化写入逻辑 ---
                if len(images) > 0:
                    # 【修复】让临时文件也以 .npz 结尾，防止 NumPy 自动追加后缀
                    tmp_path = temp_file_path.replace('.npz', '_tmp.npz')
                    np.savez_compressed(tmp_path, images=np.array(images), labels=np.array(labels))
                    os.rename(tmp_path, temp_file_path)  # 只有写完重命名才算成功
                    print(f"    [成功] 数据已保存至缓存。")
                else:
                    print(f"    [-] 警告: {base_name} 未提取到有效流量。")

            except Exception as e:
                print(f"[!] 解析 {base_name} 出错: {e}")

        return True

    def merge_and_save(self):
        """合并缓存并划分数据集，完成后清理冗余缓存"""
        print(f"\n[*] 正在合并 {self.dataset_name} 的临时缓存文件...")
        temp_files = glob.glob(os.path.join(self.temp_dir, '*.npz'))

        if not temp_files:
            print(f"[!] 错误：找不到任何缓存文件。")
            return

        all_images, all_labels = [], []
        for t_file in temp_files:
            try:
                data = np.load(t_file, allow_pickle=True)
                all_images.append(data['images'])
                all_labels.append(data['labels'])
            except Exception as e:
                print(f"    [!] 加载缓存 {t_file} 失败（可能是损坏的文件），已跳过。")

        if not all_images: return

        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # 过滤低频样本
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= 2]
        valid_indices = np.isin(labels, valid_labels)
        images, labels = images[valid_indices], labels[valid_indices]

        # ==== 【修复隐患 1】：防崩溃保护，如果只有 1 个类别，取消分层抽样 ====
        if len(valid_labels) < 2:
            print("    [!] 注意：当前提取到的流量类别少于 2 种，自动取消分层抽样 (Stratify)。")
            # 如果样本极端少（<=1个），直接全部作为测试/训练集防止报错
            if len(images) <= 1:
                X_train, X_test, y_train, y_test = np.array(images), np.array(images), np.array(labels), np.array(labels)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    images, labels, test_size=0.1, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                images, labels, test_size=0.1, random_state=42, stratify=labels
            )

        train_path = self.output_idx_path.replace('.npz', '_train.npz')
        test_path = self.output_idx_path.replace('.npz', '_test.npz')

        # 记录全局类别名单
        np.savez_compressed(train_path, images=X_train, labels=y_train, classes=valid_labels)
        np.savez_compressed(test_path, images=X_test, labels=y_test, classes=valid_labels)

        print(f"[+] 最终数据集已保存。")

        # 清理冗余中间数据
        try:
            print(f"[*] 正在清理临时缓存目录: {self.temp_dir} ...")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"[*] 清理完成，磁盘空间已释放。")
        except Exception as e:
            print(f"[!] 清理缓存目录失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetVision Preprocessing")
    parser.add_argument('--dataset', type=str, default='USTC_TFC2016',
                        help='选择要处理的数据集名称')
    parser.add_argument('--max_packets', type=int, default=300000)
    args = parser.parse_args()

    BASE_PROCESSED_DIR = "data/processed"
    datasets_to_process = ['USTC_TFC2016', 'CIC_IoT_2023', 'ToN-IoT'] if args.dataset == 'all' else [args.dataset]

    for ds_name in datasets_to_process:
        raw_data_dir = os.path.join("data/raw", ds_name)
        if not os.path.exists(raw_data_dir): continue

        safe_ds_name = ds_name.lower().replace('-', '_')
        output_path = os.path.join(BASE_PROCESSED_DIR, f"{safe_ds_name}_dataset.npz")

        preprocessor = NetVisionPreprocessor(raw_data_dir, output_path, dataset_name=ds_name,
                                             max_packets=args.max_packets)
        if preprocessor.process_all_pcaps():
            preprocessor.merge_and_save()