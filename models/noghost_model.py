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
# 2. 基础组件：通道重排与 Ghost 模块 (原版保留对比)
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
# 3. [新增] 消融实验组件：标准卷积模块
# ==========================================
class StandardConvModule(nn.Module):
    """
    用于替代 GhostModule 的标准卷积模块 (Conv2d + BatchNorm + ReLU)
    接口与 GhostModule 保持完全一致，以实现无缝替换
    """
    def __init__(self, inp, oup, kernel_size=1, stride=1, relu=True):
        super(StandardConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        return self.conv(x)


# ==========================================
# 4. 核心模块：无 Ghost 的 LRBBlock (消融版)
# ==========================================
class LRBBlock_NoGhost(nn.Module):
    """
    保留了 LightGuard 的通道拆分 (Channel Split)、通道重排 (Channel Shuffle)
    和 SE 注意力机制，但将其中的特征提取算子从 GhostModule 退化为传统的 StandardConvModule。
    """
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super(LRBBlock_NoGhost, self).__init__()
        self.stride = stride

        new_out_channel = out_chs // 2

        if stride == 1:
            new_channel = in_chs // 2
            self.branch2 = nn.Sequential(
                StandardConvModule(new_channel, mid_chs, relu=True),
                StandardConvModule(mid_chs, new_out_channel, relu=False)
            )
            self.shortcut = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )

            self.branch2 = nn.Sequential(
                StandardConvModule(in_chs, mid_chs, relu=True),
                nn.Conv2d(mid_chs, mid_chs, 3, stride=stride, padding=1, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
                StandardConvModule(mid_chs, new_out_channel, relu=False)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(new_out_channel)
            )

        self.se = SELayer(out_chs)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2) + self.shortcut(x2)), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x) + self.shortcut(x)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        return self.se(out)


# ==========================================
# 5. 主网络：NetVision (应用消融版 LRB)
# ==========================================
class NetVision(nn.Module):
    def __init__(self, num_classes=8):
        super(NetVision, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 这里全面替换为无 Ghost 的 LRB 块 (LRBBlock_NoGhost)
        self.layer1 = nn.Sequential(
            LRBBlock_NoGhost(16, 32, 32, stride=2),
            LRBBlock_NoGhost(32, 48, 32, stride=1)
        )

        self.layer2 = nn.Sequential(
            LRBBlock_NoGhost(32, 64, 64, stride=2),
            LRBBlock_NoGhost(64, 96, 64, stride=1)
        )

        self.layer3 = nn.Sequential(
            LRBBlock_NoGhost(64, 128, 128, stride=2),
            LRBBlock_NoGhost(128, 192, 128, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.f1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

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
    summary(model, (1, 28, 28))