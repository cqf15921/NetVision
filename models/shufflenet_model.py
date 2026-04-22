import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, in_chs, out_chs, stride=1):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        branch_features = out_chs // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, 3, stride, 1, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, branch_features, 1, 1, 0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        mid_chs = branch_features if self.stride > 1 else in_chs // 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(mid_chs, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride, 1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes=8):
        super(ShuffleNetV2, self).__init__()

        # 初始特征层 (1x28x28 -> 16x14x14)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer1 = nn.Sequential(
            ShuffleV2Block(16, 32, stride=2),  # 14x14 -> 7x7
            ShuffleV2Block(32, 32, stride=1)
        )
        self.layer2 = nn.Sequential(
            ShuffleV2Block(32, 64, stride=2),  # 7x7 -> 4x4
            ShuffleV2Block(64, 64, stride=1)
        )
        self.layer3 = nn.Sequential(
            ShuffleV2Block(64, 128, stride=2),  # 4x4 -> 2x2
            ShuffleV2Block(128, 128, stride=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
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
        x = self.classifier(x)
        return x