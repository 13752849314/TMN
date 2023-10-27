import torch
from torch import nn

from loss.structure import structure


class STLoss(nn.Module):
    def __init__(self, weight=1.):
        super().__init__()
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        loss = self.l1(x, y) + self.weight * (1 - structure(x, y))
        return loss


if __name__ == '__main__':
    a = torch.randn(3, 3, 24, 24).clamp(0, 1)
    b = torch.randn(3, 3, 24, 24).clamp(0, 1)
    c = STLoss()
    re = c(a, b)
    print(re)
