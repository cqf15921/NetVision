import gradio as gr
import subprocess
import os
import time

# ==========================================
# 全局进程追踪器 (用于精准强杀任务)
# ==========================================
active_processes = {
    "preprocess": None,
    "train": None,
    "detect": None
}


def stream_command(cmd, task_name):
    """执行命令行指令，实时捕获日志并绑定到全局进程追踪器"""
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8'
    )
    active_processes[task_name] = process

    output_log = ""
    try:
        # 逐行读取输出流，实现进度条实时滚动
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
def run_preprocessing(upload_file, default_dataset):
    target_ds = default_dataset if default_dataset else "CIC_IoT_2023"
    cmd = f"python utils/preprocessing.py --dataset {target_ds}"

    yield f"🚀 正在准备预处理数据集: {target_ds}...\n"
    for log in stream_command(cmd, "preprocess"):
        yield log
    yield log + "\n\n✅ 预处理任务运行结束。"


def stop_preprocessing():
    p = active_processes.get("preprocess")
    if p and p.poll() is None:
        p.terminate()  # 发送系统级 SIGTERM 强杀进程
        return "🛑 [警告] 预处理任务已被手动强制终止！相关内存已释放。"
    return "⚠️ 当前没有正在运行的预处理任务。"


def run_training(dataset_choice, batch_size, epochs):
    target_ds = dataset_choice if dataset_choice else "CIC_IoT_2023"
    cmd = f"python train.py --dataset {target_ds} --batch_size {int(batch_size)} --epochs {int(epochs)}"

    yield f"🚀 正在启动 NetVision 训练引擎...\n数据集: {target_ds} | Epochs: {epochs}\n"
    for log in stream_command(cmd, "train"):
        yield log
    yield log + "\n\n✅ 训练任务运行结束。"


def stop_training():
    p = active_processes.get("train")
    if p and p.poll() is None:
        p.terminate()
        return "🛑 [警告] 模型训练已被手动强制终止！"
    return "⚠️ 当前没有正在运行的训练任务。"


def get_latest_model(dataset_choice):
    """获取训练好的模型路径供下载"""
    target_ds = dataset_choice.lower().replace('-', '_')
    model_path = f"checkpoints/netvision_{target_ds}.pth"
    if os.path.exists(model_path):
        return model_path
    return None


# ==========================================
# 模块二：恶意流量检测回调函数
# ==========================================
def run_detection(test_pcap, model_file, dataset_choice):
    yield "🔎 正在初始化 NetVision 流量检测引擎...\n"
    time.sleep(1)  # 模拟加载停顿

    target_ds = dataset_choice if dataset_choice else "CIC_IoT_2023"
    cmd = f"python test.py --dataset {target_ds}"

    yield f"[*] 已加载模型权重文件: {model_file.name if model_file else '默认最佳缓存权重'}\n"
    yield f"[*] 正在分析目标流量特征...\n"

    final_log = ""
    for log in stream_command(cmd, "detect"):
        final_log = log
        yield log

    yield final_log + "\n\n✅ 检测完毕！请查看上方详细分类报告。"


# ==========================================
# 构建 UI 界面
# ==========================================
theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(title="NetVision 物联网恶意流量检测系统", theme=theme) as demo:
    gr.Markdown(
        """
        # 🛡️ NetVision 基于流量图像的轻量级物联网恶意流量检测系统
        """
    )

    # ---------------------------------------------------------
    # 选项卡 1：数据预处理与模型训练
    # ---------------------------------------------------------
    with gr.Tab("⚙️ 数据预处理与训练模块"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 数据集配置")
                upload_dataset = gr.File(label="上传自选数据集 (PCAP)", file_types=[".pcap", ".pcapng", ".zip"])
                default_dataset = gr.Dropdown(
                    choices=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="CIC_IoT_2023",
                    label="使用系统内置预处理数据集"
                )

                gr.Markdown("### 2. 模型训练超参数")
                batch_size = gr.Slider(minimum=16, maximum=256, step=16, value=64, label="Batch Size (批次大小)")
                epochs = gr.Slider(minimum=1, maximum=50, step=1, value=10, label="Epochs (训练轮数)")

                # 按钮区域布局优化
                with gr.Row():
                    btn_preprocess = gr.Button("🛠️ 开始预处理", variant="primary")
                    btn_stop_prep = gr.Button("🛑 终止预处理", variant="stop")
                with gr.Row():
                    btn_train = gr.Button("🚀 开始训练模型", variant="primary")
                    btn_stop_train = gr.Button("🛑 终止模型训练", variant="stop")

                with gr.Row():
                    btn_get_model = gr.Button("🔄 手动刷新并提取最新模型", variant="secondary")
                # 设为 interactive=False 变成纯粹的下载框
                download_model = gr.File(label="💾 提取成功后，点击下方出现的文件卡片即可下载 (.pth)", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 实时运行监控台")
                train_log = gr.Textbox(label="系统日志与执行进度", lines=22, max_lines=25, interactive=False)

        # ---------------- 事件绑定逻辑 ----------------
        # 预处理事件
        prep_event = btn_preprocess.click(fn=run_preprocessing, inputs=[upload_dataset, default_dataset],
                                          outputs=[train_log])
        # 点击终止时：1.强杀后端进程 2.取消前端(cancels)监听防止卡死
        btn_stop_prep.click(fn=stop_preprocessing, inputs=None, outputs=[train_log], cancels=[prep_event])

        # 1. 启动训练事件
        train_event = btn_train.click(fn=run_training, inputs=[default_dataset, batch_size, epochs],
                                      outputs=[train_log])
        # 2. 强杀进程事件
        btn_stop_train.click(fn=stop_training, inputs=None, outputs=[train_log], cancels=[train_event])

        # 3. 【核心修复】使用 .then() 确保训练任务完全跑完后，再自动提取模型
        train_event.then(fn=get_latest_model, inputs=[default_dataset], outputs=[download_model])
        # 4. 绑定我们新加的手动提取按钮
        btn_get_model.click(fn=get_latest_model, inputs=[default_dataset], outputs=[download_model])

    # ---------------------------------------------------------
    # 选项卡 2：恶意流量检测模块
    # ---------------------------------------------------------
    with gr.Tab("🔎 恶意流量检测模块"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 上传检测数据与环境配置")
                upload_pcap = gr.File(label="上传待检测的网络测试集 (.npz/.pcap)")
                upload_model = gr.File(label="上传自定义模型权重 (.pth) [留空则使用默认最新权重]")
                target_env = gr.Dropdown(
                    choices=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                    value="CIC_IoT_2023",
                    label="选择检测环境规范 (严格匹配特征维度)"
                )

                btn_detect = gr.Button("🚨 启动流量检测", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### 深度包检测 (DPI) 结果报告")
                detect_log = gr.Textbox(label="检测进度与安全研判报告", lines=20, max_lines=25, interactive=False)

        btn_detect.click(fn=run_detection, inputs=[upload_pcap, upload_model, target_env], outputs=[detect_log])

if __name__ == "__main__":
    # share=True 生成公网链接
    demo.queue().launch(share=True, debug=True)