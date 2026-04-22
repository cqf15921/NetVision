import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN1D, self).__init__()

        # 针对 1x784 一维序列的特征提取
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),  # 784 -> 196

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),  # 196 -> 49

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)  # 聚合为 128x1
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x 的输入形状是 (B, 1, 28, 28)
        B = x.size(0)
        # 将二维特征图强制展平为一维序列: (B, 1, 784)
        x = x.view(B, 1, -1)

        x = self.features(x)
        x = x.view(B, -1)
        x = self.classifier(x)
        return x