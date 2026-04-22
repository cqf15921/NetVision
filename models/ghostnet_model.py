import torch
import torch.nn as nn
# 引入您已经写好的 GhostModule
from models.netvision_model import GhostModule


class GhostBottleneck(nn.Module):
    """标准的 GhostNet 瓶颈层，用于和 NetVision 的 LRBBlock 对比"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 1. 扩展通道 (Pointwise)
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # 2. 深度卷积下采样 (Depthwise)
        if stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # 3. 降维输出 (Pointwise)
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # 残差连接设计
        if stride == 1 and in_chs == out_chs:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        x = self.ghost2(x)
        return x + residual


class GhostNet(nn.Module):
    def __init__(self, num_classes=8):
        super(GhostNet, self).__init__()

        # 初始特征层 (1x28x28 -> 16x14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 堆叠 GhostBottleneck
        self.layer1 = nn.Sequential(
            GhostBottleneck(16, 32, 24, stride=2),  # 14x14 -> 7x7
            GhostBottleneck(24, 48, 24, stride=1)
        )
        self.layer2 = nn.Sequential(
            GhostBottleneck(24, 64, 40, stride=2),  # 7x7 -> 4x4
            GhostBottleneck(40, 80, 40, stride=1)
        )
        self.layer3 = nn.Sequential(
            GhostBottleneck(40, 128, 80, stride=2),  # 4x4 -> 2x2
            GhostBottleneck(80, 160, 80, stride=1)
        )

        # 全局池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x