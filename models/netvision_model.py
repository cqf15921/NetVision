import torch
import torch.nn as nn
import math


# ==========================================
# 1. 新增：SE 注意力模块 (Squeeze-and-Excitation)
# ==========================================
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
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
def channel_shuffle(x, groups):
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
# 3. 核心模块：LRBBlock (改进版)
# ==========================================
class LRBBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super(LRBBlock, self).__init__()
        self.stride = stride

        # 将通道分为两半，一半恒等映射，一半进行特征提取
        new_channel = in_chs // 2
        new_out_channel = out_chs // 2

        if stride == 1:
            # Stride 1: 特征提取分支
            self.branch2 = nn.Sequential(
                GhostModule(new_channel, mid_chs, relu=True),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )
            # 残差连接 (对应 branch2)
            self.shortcut = nn.Sequential()
        else:
            # Stride 2: 下采样分支 (核心修改：使用 padding=1 的 AvgPool2d)
            # 分支 1：通过池化降低尺寸，不使用插值
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, 3, stride=stride, padding=1, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(new_out_channel),
                nn.ReLU(inplace=True)
            )

            # 分支 2：深度特征提取 + 下采样
            self.branch2 = nn.Sequential(
                GhostModule(in_chs, mid_chs, relu=True),
                nn.Conv2d(mid_chs, mid_chs, 3, stride=stride, padding=1, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )

            # Stride 2 时的残差映射
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(new_out_channel)
            )

        # 在 LRB 输出后集成 SE 模块 [创新点]
        self.se = SELayer(out_chs)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            # 仅对一半通道执行处理，并与另一半拼接
            out = torch.cat((x1, self.branch2(x2) + self.shortcut(x2)), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x) + self.shortcut(x)
            # 由于使用了 padding=1 的 AvgPool/Conv，out1 和 out2 尺寸天然一致，不再需要插值
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        # 最后应用注意力权重
        return self.se(out)


# ==========================================
# 4. 主网络：NetVision
# ==========================================
class NetVision(nn.Module):
    def __init__(self, num_classes=10):
        super(NetVision, self).__init__()

        # 初始特征层 (28x28 -> 14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 核心 LRB 堆叠层
        self.layer1 = nn.Sequential(
            LRBBlock(16, 32, 32, stride=1),
            LRBBlock(32, 48, 48, stride=1)
        )

        self.layer2 = nn.Sequential(
            LRBBlock(48, 96, 96, stride=2),  # 14x14 -> 7x7
            LRBBlock(96, 128, 128, stride=1)
        )

        self.layer3 = nn.Sequential(
            LRBBlock(128, 192, 192, stride=2),  # 7x7 -> 4x4
            LRBBlock(192, 256, 256, stride=1)
        )

        # 全局池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

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
    # 测试模型输入输出
    model = NetVision(num_classes=10)
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1000:.2f} K")