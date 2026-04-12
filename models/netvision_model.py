import math

import torch
import torch.nn as nn
from tensorboard.backend.event_processing.event_file_inspector import PRINT_SEPARATOR
from torch import Tensor
from torchsummary import summary


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class GhostModule(nn.Module):
    # ratio: 压缩比，用于确定原始特征图（init_channels）的数量。oup 除以 ratio 得到 init_channels。
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)  # math.ceil(x) 方法将 x 向上取整
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels=inp, out_channels=init_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding=dw_size // 2, groups=init_channels, bias=False),
            # groups 分组卷积
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class LRBBlock(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, stride=1):
        super(LRBBlock, self).__init__()
        self.stride = stride
        new_channel = in_chs // 2
        new_out_channel = out_chs // 2

        if stride == 1:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                GhostModule(new_channel, mid_chs, relu=True),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )
            self.shortcut = nn.Sequential()
        else:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.branch2 = nn.Sequential(
                GhostModule(in_chs, mid_chs, relu=True),
                nn.Conv2d(mid_chs, mid_chs, 3, stride=stride, padding=1, groups=mid_chs, bias=False),
                nn.BatchNorm2d(mid_chs),
                GhostModule(mid_chs, new_out_channel, relu=False)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, new_out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(new_out_channel)
            )

    def forward(self, x):
        if self.stride == 1:
            ranch1, ranch2 = x.chunk(2, dim=1)
            out = torch.cat((ranch1, self.branch2(ranch2) + self.shortcut(ranch2)), dim=1)
        else:
            # out = torch.cat((self.branch1(x), self.branch2(x) + self.shortcut(x)), dim=1)
            out1 = self.branch1(x)  # 池化层输出
            out2 = self.branch2(x) + self.shortcut(x)  # 卷积层输出
            out1 = nn.functional.interpolate(out1, size=out2.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat((out1, out2), dim=1)

        out = channel_shuffle(out, 2)
        return out


class NetVision(nn.Module):
    def __init__(self):
        super(NetVision, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.LRB1 = LRBBlock(32, 16, 32, 1)
        self.LRB2 = LRBBlock(32, 16, 32, 1)
        self.LRB3 = LRBBlock(32, 128, 64, 2)
        self.LRB4 = LRBBlock(64, 96, 64, 1)
        self.LRB5 = LRBBlock(64, 256, 128, 2)
        self.LRB6 = LRBBlock(128, 192, 128, 1)
        self.LRB7 = LRBBlock(128, 512, 256, 2)
        self.LRB8 = LRBBlock(256, 512, 256, 1)

        self.f1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 8)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.LRB1(x)
        x = self.LRB2(x)
        x = self.LRB3(x)
        x = self.LRB4(x)
        x = self.LRB5(x)
        x = self.LRB6(x)
        x = self.LRB7(x)
        x = self.LRB8(x)
        y = self.f1(x)
        return y

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NetVision().to(device)
    # 将测试输入尺寸改为 (1, 28, 28) 以匹配 784 字节的截断长度
    print(summary(model, (1, 28, 28)))