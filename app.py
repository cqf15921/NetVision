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
    if default_dataset == "无":
        if upload_dir is None:
            yield "❌ 错误：选择了【无】规范时，必须上传包含流量包的文件夹！\n"
            return
        target_ds = "User_Dataset"  # 为用户自定义上传的数据集分配统一内部名称

        # ==== 【修复隐患 2】：清空旧的自定义数据和缓存，防止数据无限叠加和同名文件被跳过 ====
        raw_dir = f"data/raw/{target_ds}"
        temp_cache = f"data/processed/temp_{target_ds}"
        shutil.rmtree(raw_dir, ignore_errors=True)
        shutil.rmtree(temp_cache, ignore_errors=True)
    else:
        target_ds = default_dataset

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
    yield f"🚀 正在启动预处理引擎 (目标数据集: {target_ds}, 采样上限: {int(max_packets) if max_packets > 0 else '全量读取'})...\n"
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
    target_ds = "User_Dataset" if dataset_choice == "无" else dataset_choice
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
    target_ds_name = "User_Dataset" if dataset_choice == "无" else dataset_choice
    target_ds = target_ds_name.lower().replace('-', '_')
    model_path = f"checkpoints/netvision_{target_ds}.pth"
    if os.path.exists(model_path):
        return model_path
    return None


# ==========================================
# 模块二：恶意流量检测回调函数
# ==========================================
def run_detection(test_file, model_file, dataset_choice):
    """执行检测，并确保可视化结果实时刷新"""

    # 启动前清理旧的评估结果图片，防止旧图残留误导用户
    for img_file in ["results/confusion_matrix.png", "results/metrics_bar.png"]:
        if os.path.exists(img_file):
            try:
                os.remove(img_file)
            except Exception:
                pass

    yield "🔎 正在初始化 NetVision 流量检测引擎...\n", None, None
    time.sleep(1)

    # 处理“无”选项（自定义检测模式）
    if dataset_choice == "无":
        if not test_file or not model_file:
            yield "❌ 错误：选择了【无】规范时，必须上传 [待检测特征集 .npz] 和 [自定义权重 .pth]！\n", None, None
            return

        test_path = getattr(test_file, 'name', str(test_file))
        model_path = getattr(model_file, 'name', str(model_file))

        # 组装带自定义路径的命令
        cmd = f"python test.py --custom_test_path \"{test_path}\" --custom_model_path \"{model_path}\""

        yield f"[*] 模式: 自定义检测\n[*] 特征集路径: {test_path}\n[*] 权重路径: {model_path}\n", None, None
    else:
        # 使用系统内置的数据集和模型
        cmd = f"python test.py --dataset {dataset_choice}"
        yield f"[*] 模式: 内置规范检测\n[*] 检测规范: {dataset_choice}\n", None, None

    yield f"[*] 正在分析目标流量特征并生成报告...\n", None, None

    final_log = ""
    for log in stream_command(cmd, "detect"):
        final_log = log
        # 实时流输出时，图片占位为 None
        yield log, None, None

    # 检测结束后，读取 test.py 生成的图片文件路径
    cm_img = "results/confusion_matrix.png" if os.path.exists("results/confusion_matrix.png") else None
    bar_img = "results/metrics_bar.png" if os.path.exists("results/metrics_bar.png") else None

    # 将生成的图片路径传给前端组件
    yield final_log + "\n\n✅ 检测与可视化完毕！", bar_img, cm_img


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
                    choices=["无", "CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="无",
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
                upload_test = gr.File(label="上传待检测的流量特征集 (.npz)")
                upload_weight = gr.File(label="上传自定义权重 (.pth)")
                target_env = gr.Dropdown(
                    choices=["无", "CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="无",
                    label="选择检测规范"
                )
                btn_detect = gr.Button("🚨 启动流量检测识别", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 识别结果报告")
                detect_log = gr.Textbox(label="检测进度与安全研判", lines=10, max_lines=15, interactive=False)

                with gr.Row():
                    metrics_plot = gr.Image(label="总体性能指标柱状图", type="filepath")
                    cm_plot = gr.Image(label="分类混淆矩阵", type="filepath")

        # 绑定检测事件
        btn_detect.click(
            fn=run_detection,
            inputs=[upload_test, upload_weight, target_env],
            outputs=[detect_log, metrics_plot, cm_plot]
        )

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)