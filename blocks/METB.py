import torch
from torch import nn

from .RCAB import RCAB
from .transformer import MLABlock

from utils.transformerUtil import reverse_patches


class METB(nn.Module):
    def __init__(self, filters, ratio=8, drop=0., kernel_size=1, stride=1, padding=0, dilation=1):
        super(METB, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size, stride, padding=padding * dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(filters * 2, filters, kernel_size, stride, padding=padding * dilation, dilation=dilation)
        self.conv3 = nn.Conv2d(filters * 2, filters, kernel_size, stride, padding=padding * dilation, dilation=dilation)
        self.conv4 = nn.Conv2d(filters * 2, filters, kernel_size, stride, padding=padding * dilation, dilation=dilation)

        self.rcab1 = RCAB(filters, filters, ratio)
        self.rcab2 = RCAB(filters, filters, ratio)

        self.tf = MLABlock(filters, dim=9 * filters, drop=drop)

    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.conv1(x)
        x2 = self.rcab1(x1)

        x3 = torch.cat([x2, x1], dim=1)
        x3 = self.conv2(x3)

        x4 = self.tf(x3)
        x4 = x4.permute(0, 2, 1)
        x4 = reverse_patches(x4, (h, w), (3, 3), 1, 1)
        x4 = torch.cat([x4, x1], dim=1)

        x4 = self.conv3(x4)
        x4 = self.rcab2(x4)

        x5 = torch.cat([x4, x1], dim=1)
        out = self.conv4(x5)
        return out


class METB_out(METB):
    def __init__(self, filters, out, ratio=8, drop=0., kernel_size=1, stride=1, padding=0, dilation=1):
        super(METB_out, self).__init__(filters, ratio, drop, kernel_size, stride, padding, dilation)
        self.exit = nn.Conv2d(filters, out, kernel_size, stride, padding=padding * dilation, dilation=dilation)

    def forward(self, x):
        out = super(METB_out, self).forward(x)
        out = self.exit(out)
        return out


if __name__ == '__main__':
    from utils.common import print_network

    model = METB(16)
    print(model)
    print_network(model)

    t = torch.randn(1, 16, 64, 64)
    re = model(t)

    print(re.shape)

    model = METB_out(16, 4)
    print(model)
    print_network(model)
    re = model(t)
    print(re.shape)
