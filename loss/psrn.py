from math import log10

import numpy as np
import torch


def mse(x, y):
    """
    MSE Loss
    :param x: tensor / numpy.ndarray
    :param y: tensor / numpy.ndarray
    :return: float
    """
    diff = x - y
    diff = diff * diff
    if isinstance(diff, np.ndarray):
        diff = torch.FloatTensor(diff)
    return torch.mean(diff)


def psnr(x, y, peak=1.):
    """
    psnr from tensor
    :param x: tensor
    :param y: tensor
    :param peak:
    :return: float
    """
    _mse = mse(x, y)
    return 10 * log10((peak ** 2) / _mse)
