import torch
import torch.nn as nn
from torchsummary import summary


class ResBlock(nn.Module):
    """
    经典残差块 (Standard Residual Block / BasicBlock)
    用于消融实验的 Baseline 模块
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut 分支 (用于维度匹配和下采样对齐)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class NetVision_Ablation(nn.Module):
    """
    NetVision 的消融实验版本
    移除了轻量化的 LRBBlock (基于 GhostModule 和 Channel Shuffle)，
    全部替换为经典的 ResBlock，以证明轻量化设计的必要性。
    """

    def __init__(self, num_classes=8):
        super(NetVision_Ablation, self).__init__()
        # 初始卷积层保持不变
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 使用经典 ResBlock 完全替换原有的 LRBBlock
        # 参数对齐原始架构的输入/输出通道数以及下采样步长
        self.layer1 = ResBlock(in_channels=32, out_channels=32, stride=1)
        self.layer2 = ResBlock(in_channels=32, out_channels=32, stride=1)
        self.layer3 = ResBlock(in_channels=32, out_channels=64, stride=2)
        self.layer4 = ResBlock(in_channels=64, out_channels=64, stride=1)
        self.layer5 = ResBlock(in_channels=64, out_channels=128, stride=2)
        self.layer6 = ResBlock(in_channels=128, out_channels=128, stride=1)
        self.layer7 = ResBlock(in_channels=128, out_channels=256, stride=2)
        self.layer8 = ResBlock(in_channels=256, out_channels=256, stride=1)

        # 分类头保持不变
        self.f1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        y = self.f1(x)
        return y


if __name__ == "__main__":
    # 测试代码：实例化模型并打印结构和参数量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 假设输入为单通道 (例如灰度图或转换后的流量特征图)，尺寸 28x28，分类数为 8
    model = NetVision_Ablation(num_classes=8).to(device)

    print("=" * 50)
    print("Ablation Model (Classic ResBlock) Summary")
    print("=" * 50)
    print(summary(model, (1, 28, 28)))