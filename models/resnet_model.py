import torch
import torch.nn as nn
import math
from torch import Tensor


# ==========================================
# 1. SE 注意力模块 (Squeeze-and-Excitation)
# ==========================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 降维比例 reduction 确保计算轻量化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==========================================
# 2. 基础组件：通道重排与 Ghost 模块
# ==========================================
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# ==========================================
# 3. 核心模块：LRBBlock (原版轻量级残差块，保留以备后用)
# ==========================================
class LRBBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super(LRBBlock, self).__init__()
        self.stride = stride

        # 左右分支的输出通道各占总输出的一半
        new_out_channel = out_chs // 2

        if stride == 1:
            # 此时要求 in_chs == out_chs
            new_channel = in_chs // 2
            self.branch2 = nn.Sequential(
                GhostModule(new_channel, mid_chs, relu=True),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )
            self.shortcut = nn.Sequential()
        else:
            # 下采样分支：16 -> 32
            # 分支 1：使用 AvgPool2d 代替插值，保证边缘信息完整
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )

            # 分支 2：深度特征提取 + 下采样
            self.branch2 = nn.Sequential(
                GhostModule(in_chs, mid_chs, relu=True),
                nn.Conv2d(mid_chs, mid_chs, 3, stride=stride, padding=1, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )

            # 尺寸减半时的残差映射
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(new_out_channel)
            )

        # 在每个 LRB 块末尾集成 SE 注意力模块
        self.se = SELayer(out_chs)

    def forward(self, x):
        if self.stride == 1:
            # Stride=1 使用 Channel Split
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2) + self.shortcut(x2)), dim=1)
        else:
            # Stride=2 左右分支直接拼接，完成通道翻倍
            out1 = self.branch1(x)
            out2 = self.branch2(x) + self.shortcut(x)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        return self.se(out)


# ==========================================
# 新增：用于消融实验的经典残差块 (Classical Residual Block)
# ==========================================
class ClassicalResidualBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        # 注意：标准残差块不需要 mid_chs，但为了保持与原来 NetVision 传参格式的兼容性，保留该参数
        super(ClassicalResidualBlock, self).__init__()

        # 第一个 3x3 标准卷积 (如果 stride=2 则在这里下采样)
        self.conv1 = nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)

        # 第二个 3x3 标准卷积
        self.conv2 = nn.Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chs)

        self.relu = nn.ReLU(inplace=True)

        # 捷径分支 (Shortcut connection)
        self.shortcut = nn.Sequential()
        # 如果通道数发生变化，或者步长不为 1，需要使用 1x1 卷积调整维度以确保可以相加
        if stride != 1 or in_chs != out_chs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chs)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        out += identity
        out = self.relu(out)

        return out


# ==========================================
# 4. 主网络：NetVision (当前被修改为消融实验版本)
# ==========================================
class NetVision(nn.Module):
    def __init__(self, num_classes=8):
        super(NetVision, self).__init__()

        # 初始特征层 (28x28 -> 14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 替换为经典残差块 (ClassicalResidualBlock) 用于消融实验
        # 若要恢复轻量级网络，将下面的 ClassicalResidualBlock 改回 LRBBlock 即可
        self.layer1 = nn.Sequential(
            ClassicalResidualBlock(16, 32, 32, stride=2),  # 14x14 -> 7x7, 16 -> 32
            ClassicalResidualBlock(32, 48, 32, stride=1)
        )

        self.layer2 = nn.Sequential(
            ClassicalResidualBlock(32, 64, 64, stride=2),  # 7x7 -> 4x4, 32 -> 64
            ClassicalResidualBlock(64, 96, 64, stride=1)
        )

        self.layer3 = nn.Sequential(
            ClassicalResidualBlock(64, 128, 128, stride=2),  # 4x4 -> 2x2, 64 -> 128
            ClassicalResidualBlock(128, 192, 128, stride=1)
        )

        # 全局池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Sequential(
            nn.Linear(128, 256),  # 输入维度需匹配 layer3 的输出 128
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetVision(num_classes=8).to(device)
    # 输入为 (1, 28, 28) 以匹配 784 字节截断长度
    summary(model, (1, 28, 28))