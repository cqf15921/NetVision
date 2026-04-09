import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import subprocess
import threading
import os


class NetVisionDesktopApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NetVision 物联网恶意流量检测系统 (Desktop版)")
        self.root.geometry("900x700")
        self.root.configure(padx=10, pady=10)

        # 进程追踪器
        self.processes = {"prep": None, "train": None, "detect": None}

        # 样式设置
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('微软雅黑', 10), padding=5)
        style.configure('TLabel', font=('微软雅黑', 10))
        style.configure('Header.TLabel', font=('微软雅黑', 14, 'bold'), foreground='darkblue')

        # 顶部标题
        ttk.Label(self.root, text="🛡️ NetVision 深度学习物联网恶意流量检测系统", style='Header.TLabel').pack(
            pady=(0, 10))

        # 创建选项卡
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_train = ttk.Frame(self.notebook, padding=10)
        self.tab_detect = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.tab_train, text="⚙️ 数据预处理与训练")
        self.notebook.add(self.tab_detect, text="🔎 恶意流量检测")

        self.build_train_tab()
        self.build_detect_tab()

    # ==========================================
    # 核心机制：多线程执行命令与日志捕获
    # ==========================================
    def run_command(self, cmd, task_name, log_widget, end_msg="✅ 任务运行结束。"):
        def target():
            log_widget.insert(tk.END, f"\n执行命令: {cmd}\n")
            log_widget.insert(tk.END, "-" * 50 + "\n")
            log_widget.see(tk.END)

            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                errors='replace'
            )
            self.processes[task_name] = process

            try:
                for line in iter(process.stdout.readline, ''):
                    log_widget.insert(tk.END, line)
                    log_widget.see(tk.END)  # 自动滚动到最底部
            finally:
                process.stdout.close()
                process.wait()
                self.processes[task_name] = None

                # 如果是正常结束（未被手动强杀）
                if process.returncode == 0:
                    log_widget.insert(tk.END, f"\n{end_msg}\n")
                else:
                    log_widget.insert(tk.END, f"\n🛑 任务已终止或发生错误 (返回码: {process.returncode})\n")
                log_widget.see(tk.END)

        # 启动后台线程，防止 GUI 界面卡死
        threading.Thread(target=target, daemon=True).start()

    def stop_command(self, task_name, log_widget):
        p = self.processes.get(task_name)
        if p and p.poll() is None:
            p.terminate()
            log_widget.insert(tk.END, "\n[警告] 收到终止信号，正在强行结束进程...\n")
            log_widget.see(tk.END)
        else:
            messagebox.showinfo("提示", "当前没有正在运行的该项任务。")

    # ==========================================
    # 构建：预处理与训练模块
    # ==========================================
    def build_train_tab(self):
        # 左侧控制面板
        control_frame = ttk.Frame(self.tab_train, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 数据集选择
        ttk.Label(control_frame, text="1. 数据集配置", font=('微软雅黑', 11, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.train_ds_var = tk.StringVar(value="CIC_IoT_2023")
        ttk.Combobox(control_frame, textvariable=self.train_ds_var, values=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                     state="readonly").pack(fill=tk.X, pady=(0, 15))

        # 预处理按钮
        frame_prep_btns = ttk.Frame(control_frame)
        frame_prep_btns.pack(fill=tk.X, pady=(0, 20))
        ttk.Button(frame_prep_btns, text="🛠️ 开始预处理", command=self.start_prep).pack(side=tk.LEFT, expand=True,
                                                                                        fill=tk.X, padx=(0, 2))
        ttk.Button(frame_prep_btns, text="🛑 终止预处理",
                   command=lambda: self.stop_command("prep", self.train_log)).pack(side=tk.LEFT, expand=True, fill=tk.X,
                                                                                   padx=(2, 0))

        # 训练参数设置
        ttk.Label(control_frame, text="2. 模型训练超参数", font=('微软雅黑', 11, 'bold')).pack(anchor=tk.W, pady=(0, 5))

        ttk.Label(control_frame, text="Batch Size:").pack(anchor=tk.W)
        self.batch_var = tk.StringVar(value="64")
        ttk.Spinbox(control_frame, from_=16, to=256, increment=16, textvariable=self.batch_var).pack(fill=tk.X,
                                                                                                     pady=(0, 5))

        ttk.Label(control_frame, text="Epochs:").pack(anchor=tk.W)
        self.epoch_var = tk.StringVar(value="10")
        ttk.Spinbox(control_frame, from_=1, to=50, increment=1, textvariable=self.epoch_var).pack(fill=tk.X,
                                                                                                  pady=(0, 15))

        # 训练按钮
        frame_train_btns = ttk.Frame(control_frame)
        frame_train_btns.pack(fill=tk.X)
        ttk.Button(frame_train_btns, text="🚀 开始训练模型", command=self.start_train).pack(side=tk.LEFT, expand=True,
                                                                                           fill=tk.X, padx=(0, 2))
        ttk.Button(frame_train_btns, text="🛑 终止模型训练",
                   command=lambda: self.stop_command("train", self.train_log)).pack(side=tk.LEFT, expand=True,
                                                                                    fill=tk.X, padx=(2, 0))

        # 右侧日志面板
        log_frame = ttk.LabelFrame(self.tab_train, text=" 实时运行监控台 ")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.train_log = scrolledtext.ScrolledText(log_frame, bg="black", fg="lightgreen", font=("Consolas", 10))
        self.train_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.train_log.insert(tk.END, "系统初始化就绪...\n")

    # ==========================================
    # 构建：恶意流量检测模块
    # ==========================================
    def build_detect_tab(self):
        control_frame = ttk.Frame(self.tab_detect, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(control_frame, text="1. 环境规范 (匹配特征数)", font=('微软雅黑', 11, 'bold')).pack(anchor=tk.W,
                                                                                                      pady=(0, 5))
        self.detect_ds_var = tk.StringVar(value="CIC_IoT_2023")
        ttk.Combobox(control_frame, textvariable=self.detect_ds_var, values=["CIC_IoT_2023", "USTC_TFC2016", "ToN-IoT"],
                     state="readonly").pack(fill=tk.X, pady=(0, 15))

        ttk.Label(control_frame, text="2. 模型权重文件 (.pth)", font=('微软雅黑', 11, 'bold')).pack(anchor=tk.W,
                                                                                                    pady=(0, 5))
        self.model_path_var = tk.StringVar(value="默认最新权重")

        frame_model = ttk.Frame(control_frame)
        frame_model.pack(fill=tk.X, pady=(0, 15))
        ttk.Entry(frame_model, textvariable=self.model_path_var, state="readonly").pack(side=tk.LEFT, fill=tk.X,
                                                                                        expand=True)
        ttk.Button(frame_model, text="浏览", width=5, command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(control_frame, text="🚨 启动流量深度检测 (DPI)", command=self.start_detect).pack(fill=tk.X, pady=20)

        log_frame = ttk.LabelFrame(self.tab_detect, text=" 深度包检测与安全研判报告 ")
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.detect_log = scrolledtext.ScrolledText(log_frame, bg="#1E1E1E", fg="#00FFCC", font=("Consolas", 10))
        self.detect_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detect_log.insert(tk.END, "等待导入检测数据...\n")

    # ==========================================
    # 按钮事件绑定
    # ==========================================
    def start_prep(self):
        ds = self.train_ds_var.get()
        cmd = f"python utils/preprocessing.py --dataset {ds}"
        self.run_command(cmd, "prep", self.train_log)

    def start_train(self):
        ds = self.train_ds_var.get()
        bs = self.batch_var.get()
        ep = self.epoch_var.get()
        cmd = f"python train.py --dataset {ds} --batch_size {bs} --epochs {ep}"
        self.run_command(cmd, "train", self.train_log)

    def browse_model(self):
        path = filedialog.askopenfilename(title="选择模型权重",
                                          filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")])
        if path:
            self.model_path_var.set(path)

    def start_detect(self):
        ds = self.detect_ds_var.get()
        # 注意: 这里的 test.py 目前只接收 --dataset, 我们通过模拟的方式调用
        cmd = f"python test.py --dataset {ds}"
        self.detect_log.insert(tk.END, f"\n🔎 准备加载环境: {ds} | 权重: {self.model_path_var.get()}\n")
        self.run_command(cmd, "detect", self.detect_log, end_msg="✅ 检测分析完毕！请查看上方分类报告。")


if __name__ == "__main__":
    root = tk.Tk()
    app = NetVisionDesktopApp(root)
    root.mainloop()