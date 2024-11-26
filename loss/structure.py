import torch


def cov(x: torch.Tensor, y: torch.Tensor):
    n = x.dim()
    assert n in [4, 3, 2], f'input dim {n} not support!'
    if n == 4:
        dim = [1, 2, 3]
    else:
        dim = 0
    return torch.mean(x * y, dim=dim) - torch.mean(x, dim=dim) * torch.mean(y, dim=dim), \
        torch.std(x, dim=dim), \
        torch.std(y, dim=dim)


def structure(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape, f'x:shape {x.shape} not match y:shape {y.shape}!'
    cov_xy, std_x, std_y = cov(x, y)
    return torch.mean(cov_xy / (std_x * std_y))


if __name__ == '__main__':
    a = torch.randn(3, 3, 24, 24).clamp(0, 1)
    b = torch.randn(3, 3, 24, 24).clamp(0, 1)
    print(structure(a, b))
