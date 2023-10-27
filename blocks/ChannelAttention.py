# create by 敖鸥 at 2023/3/28
import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.conv2(self.relu1(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu1(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sig(out)


class CALayer(nn.Module):
    def __init__(self, channel, ratio=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1),
            nn.ReLU(True),
            nn.Conv2d(channel // ratio, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


if __name__ == '__main__':
    model = ChannelAttention(64)
    print(model)
    xx = torch.randn(1, 64, 256, 256)
    y = model(xx)
    print(y.shape)
