import math
import torch
import torch.nn as nn
from torchsummary import summary


def _make_divisible(v, divisor, min_value=None):
    """
    确保所有层的通道数都能被 divisor 整除
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-Excitation 模块 (SE Attention)
    """

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None, divisor=4):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = nn.Hardsigmoid()
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x_se = self.gate_fn(x_se)
        return x * x_se


class GhostModule(nn.Module):
    """
    原版 GhostModule
    """

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


class GhostBottleneck(nn.Module):
    """
    GhostNet 基础瓶颈块
    """

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # Shortcut
        if in_chs == out_chs and self.stride == 1:
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
        residual = x
        x = self.ghost1(x)
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    """
    官方 GhostNet 架构 (适配 1 通道输入)
    """

    def __init__(self, cfgs=None, num_classes=8, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()

        # 默认的 GhostNet 配置参数: [kernel_size, exp_size, out_channels, se_ratio, stride]
        if cfgs is None:
            self.cfgs = [
                [3, 16, 16, 0, 1],
                [3, 48, 24, 0, 2],
                [3, 72, 24, 0, 1],
                [5, 72, 40, 0.25, 2],
                [5, 120, 40, 0.25, 1],
                [3, 240, 80, 0, 2],
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
                [5, 672, 160, 0.25, 2],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0.25, 1]
            ]
        else:
            self.cfgs = cfgs

        output_channel = _make_divisible(16 * width, 4)

        # 针对流量特征映射的单通道输入 (in_channels=1)
        self.conv_stem = nn.Conv2d(1, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)

        # 构建 GhostBottleneck 层
        stages = []
        block = GhostBottleneck
        for k, exp_size, c, se_ratio, s in self.cfgs:
            output_channel_block = _make_divisible(c * width, 4)
            hidden_channel = _make_divisible(exp_size * width, 4)
            stages.append(block(output_channel, hidden_channel, output_channel_block, k, s, se_ratio))
            output_channel = output_channel_block
        self.blocks = nn.Sequential(*stages)

        # 构建输出层
        output_channel_conv = _make_divisible(self.cfgs[-1][2] * width, 4)
        self.conv_head = nn.Conv2d(output_channel, output_channel_conv, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel_conv)
        self.act2 = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_tail = nn.Conv2d(output_channel_conv, 1280, 1, 1, 0, bias=True)
        self.act3 = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(1280, num_classes),
        )

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.global_pool(x)
        x = self.conv_tail(x)
        x = self.act3(x)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型，使用与你的测试代码相同的输入维度与分类数
    model = GhostNet(num_classes=8).to(device)

    print("=" * 50)
    print("Official GhostNet Baseline Summary")
    print("=" * 50)
    print(summary(model, (1, 28, 28)))