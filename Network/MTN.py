import torch
from torch import nn

from blocks.METB import METB
from blocks.block import Upsampler


class MTN(nn.Module):
    def __init__(self, in_ch, out_ch, scale=4, first_filters=32, filters=16, ratio=8, drop=0.,
                 kernel_size=1, stride=1, padding=0, dilation=1
                 ):
        super(MTN, self).__init__()

        self.entry = nn.Conv2d(in_ch, first_filters, 3, 1, 1)
        self.entry_shrink = nn.Conv2d(first_filters, filters, 3, 1, 1)

        self.m11 = METB(filters, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m12 = METB(filters, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m13 = METB(filters, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m21 = METB(filters, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m22 = METB(filters * 2, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m23 = METB(filters * 3, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m31 = METB(filters, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m32 = METB(filters * 3, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)
        self.m33 = METB(filters * 6, ratio, drop, kernel_size=kernel_size, stride=stride, padding=padding,
                        dilation=dilation)

        self.fusion = nn.Conv2d(filters * 10, filters * 3, 3, 1, 1, bias=False)
        self.shrink = nn.Conv2d(filters * 3, filters, 3, 1, 1, bias=False)

        self.upsample = Upsampler(nn.Conv2d, scale, filters)

        self.exit = nn.Conv2d(filters, out_ch, 3, 1, 1, bias=False)

        self.bilinear = nn.Upsample(scale_factor=scale, mode='bilinear')

    def forward(self, x):
        x1 = self.entry(x)
        x1 = self.entry_shrink(x1)  # 1
        bilinear = self.bilinear(x)  # 2

        m11 = self.m11(x1)
        m21 = self.m21(m11)
        m31 = self.m31(m21)

        m12 = self.m12(m11)
        m13 = self.m13(m12)

        m22 = self.m22(torch.cat([m12, m21], dim=1))
        m23 = self.m23(torch.cat([m13, m22], dim=1))
        m32 = self.m32(torch.cat([m31, m22], dim=1))

        m33 = self.m33(torch.cat([m23, m32], dim=1))

        out = torch.cat([m31, m32, m33], dim=1)
        out = self.fusion(out)
        out = self.shrink(out)

        out = out + x1

        out = self.upsample(out)
        out = self.exit(out)

        return out + bilinear


if __name__ == '__main__':
    from utils.common import print_network, Flops

    model = MTN(3, 3, first_filters=32, filters=16)
    print(model)
    print_network(model)

    t = torch.randn(16, 3, 48, 48)
    re = model(t)
    Flops(model, t)
    print(re.shape)
