import torch
from torch import nn

from .ChannelAttention import CALayer


class RCAB(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=16):
        super(RCAB, self).__init__()
        modules_body = [
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            CALayer(out_ch, ratio=ratio)
        ]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        out = self.body(x)
        return out + x


if __name__ == '__main__':
    model = RCAB(32, 32)
    print(model)
    t = torch.randn(1, 32, 64, 64)
    re = model(t)
    print(re.shape)
