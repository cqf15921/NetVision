import gradio as gr
import subprocess
import os
import time
import shutil
import psutil  # 必须安装: pip install psutil

# ==========================================
# 全局进程追踪器 (用于精准强杀任务)
# ==========================================
active_processes = {
    "preprocess": None,
    "train": None,
    "detect": None
}


def kill_process_tree(pid):
    """递归强杀指定 PID 及其所有子进程，解决 Shell 模式下停不掉的问题"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()  # 强杀所有子进程（如 Python 训练进程）
        parent.kill()  # 强杀父进程（如 Shell 壳）
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"清理进程树时出错: {e}")


def stream_command(cmd, task_name):
    """执行命令行指令，实时捕获日志并绑定到全局进程追踪器"""
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )
    active_processes[task_name] = process

    output_log = ""
    try:
        for line in iter(process.stdout.readline, ''):
            output_log += line
            yield output_log
    finally:
        process.stdout.close()
        process.wait()
        active_processes[task_name] = None


# ==========================================
# 模块一：预处理与训练回调函数
# ==========================================
def run_preprocessing(upload_dir, default_dataset, max_packets):
    """执行预处理，支持文件夹上传和采样上限设置"""
    target_ds = default_dataset if default_dataset else "CIC_IoT_2023"

    if upload_dir is not None:
        raw_dir = f"data/raw/{target_ds}"
        os.makedirs(raw_dir, exist_ok=True)
        yield f"📁 检测到上传内容，正在同步文件至 {target_ds} 原始数据目录...\n"

        file_list = upload_dir if isinstance(upload_dir, list) else [upload_dir]
        valid_files = 0
        for file_obj in file_list:
            file_path = getattr(file_obj, 'name', str(file_obj))
            if file_path.lower().endswith(('.pcap', '.pcapng')):
                try:
                    shutil.copy(file_path, raw_dir)
                    valid_files += 1
                except Exception:
                    pass
        yield f"✅ 成功导入 {valid_files} 个流量包文件。\n"

    # 将采样上限参数传递给后台脚本
    cmd = f"python utils/preprocessing.py --dataset {target_ds} --max_packets {int(max_packets)}"
    yield f"🚀 正在启动预处理引擎 (采样上限: {int(max_packets) if max_packets > 0 else '全量读取'})...\n"
    for log in stream_command(cmd, "preprocess"):
        yield log
    yield log + "\n\n✅ 预处理任务结束。"


def stop_preprocessing():
    p = active_processes.get("preprocess")
    if p and p.poll() is None:
        kill_process_tree(p.pid)
        return "🛑 [强力终止] 预处理任务已强制停止，后台进程已清理。"
    return "⚠️ 当前没有正在运行的预处理任务。"


def run_training(dataset_choice, batch_size, epochs):
    target_ds = dataset_choice if dataset_choice else "CIC_IoT_2023"
    cmd = f"python train.py --dataset {target_ds} --batch_size {int(batch_size)} --epochs {int(epochs)}"

    yield f"🚀 正在启动 NetVision 训练引擎...\n数据集: {target_ds} | Epochs: {epochs}\n"
    for log in stream_command(cmd, "train"):
        yield log
    yield log + "\n\n✅ 训练任务结束。"


def stop_training():
    p = active_processes.get("train")
    if p and p.poll() is None:
        kill_process_tree(p.pid)
        return "🛑 [强力终止] 模型训练已强制停止，算力资源已释放。"
    return "⚠️ 当前没有正在运行的训练任务。"


def get_latest_model(dataset_choice):
    """获取训练生成的最佳模型权重文件"""
    target_ds = dataset_choice.lower().replace('-', '_')
    model_path = f"checkpoints/netvision_{target_ds}.pth"
    if os.path.exists(model_path):
        return model_path
    return None


# ==========================================
# 模块二：恶意流量检测回调函数
# ==========================================
def run_detection(test_file, model_file, dataset_choice):
    yield "🔎 正在初始化 NetVision 流量检测引擎...\n"
    time.sleep(1)

    target_ds = dataset_choice if dataset_choice else "CIC_IoT_2023"
    cmd = f"python test.py --dataset {target_ds}"

    yield f"[*] 检测规范: {target_ds}\n"
    yield f"[*] 正在分析目标流量特征并生成报告...\n"

    final_log = ""
    for log in stream_command(cmd, "detect"):
        final_log = log
        yield log
    yield final_log + "\n\n✅ 检测完毕！"


# ==========================================
# 构建 UI 界面
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="NetVision 物联网恶意流量检测系统", theme=theme) as demo:
    gr.Markdown(
        """
        # 🛡️ NetVision 基于流量图像的轻量级物联网恶意流量检测系统
        基于轻量级神经网络 (GhostModule & 分组卷积) 的 IoT 异常流量识别与防御平台
        """
    )

    with gr.Tab("⚙️ 数据预处理与训练模块"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 数据集配置")
                upload_dataset = gr.File(label="📂 上传包含流量包的文件夹", file_count="directory")
                default_dataset = gr.Dropdown(
                    choices=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="CIC_IoT_2023",
                    label="使用系统内置数据集"
                )

                # 采样包数控制滑动条
                max_pkts = gr.Slider(minimum=0, maximum=1000000, step=50000, value=300000,
                                     label="每个文件读取上限 (0 表示全量读取)")

                with gr.Row():
                    btn_preprocess = gr.Button("🛠️ 开始预处理", variant="primary")
                    btn_stop_prep = gr.Button("🛑 终止预处理", variant="stop")

                gr.Markdown("### 2. 模型训练超参数")
                batch_size = gr.Slider(minimum=16, maximum=256, step=16, value=64, label="Batch Size")
                epochs = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Epochs")

                with gr.Row():
                    btn_train = gr.Button("🚀 开始训练模型", variant="primary")
                    btn_stop_train = gr.Button("🛑 终止模型训练", variant="stop")

                btn_get_model = gr.Button("🔄 手动刷新并提取最新模型", variant="secondary")
                download_model = gr.File(label="💾 点击下方文件卡片下载权重 (.pth)", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 实时运行监控台")
                train_log = gr.Textbox(label="系统日志与进度", lines=22, max_lines=25, interactive=False)

        # 绑定事件逻辑
        prep_event = btn_preprocess.click(fn=run_preprocessing, inputs=[upload_dataset, default_dataset, max_pkts],
                                          outputs=[train_log])
        btn_stop_prep.click(fn=stop_preprocessing, inputs=None, outputs=[train_log], cancels=[prep_event])

        train_event = btn_train.click(fn=run_training, inputs=[default_dataset, batch_size, epochs],
                                      outputs=[train_log])
        btn_stop_train.click(fn=stop_training, inputs=None, outputs=[train_log], cancels=[train_event])

        # 链式调用确保训练后自动提取
        train_event.then(fn=get_latest_model, inputs=[default_dataset], outputs=[download_model])
        btn_get_model.click(fn=get_latest_model, inputs=[default_dataset], outputs=[download_model])

    with gr.Tab("🔎 恶意流量检测模块"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 配置检测环境")
                upload_test = gr.File(label="上传待检测测试集 (.npz/.pcap)")
                upload_weight = gr.File(label="上传自定义权重 (.pth)")
                target_env = gr.Dropdown(
                    choices=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="CIC_IoT_2023",
                    label="选择检测规范"
                )
                btn_detect = gr.Button("🚨 启动流量检测识别", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 识别结果报告")
                detect_log = gr.Textbox(label="检测进度与安全研判", lines=20, max_lines=25, interactive=False)

        btn_detect.click(fn=run_detection, inputs=[upload_test, upload_weight, target_env], outputs=[detect_log])

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)