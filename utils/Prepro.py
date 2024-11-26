import random

import torch
from PIL import Image


def random_pre_process_pair(hr: Image, lr: Image, lr_patch_size, scale):
    """
    For Real Paired Images,
    Crop hr, lr images to patches, and random pre-processing
    :param hr: PIL.Image
    :param lr: PIL.Image
    :param lr_patch_size: lr patch size
    :param scale: scale
    :return: PIL.Image
    """
    w, h = lr.size
    start_x = random.randint(0, w - lr_patch_size)
    start_y = random.randint(0, h - lr_patch_size)

    hr_patch = hr.crop((start_x * scale, start_y * scale,
                        (start_x + lr_patch_size) * scale, (start_y + lr_patch_size) * scale))
    lr_patch = lr.crop((start_x, start_y, start_x + lr_patch_size, start_y + lr_patch_size))

    if bool(random.getrandbits(1)):  # 水平
        hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
    if bool(random.getrandbits(1)):  # 垂直
        hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)
    angle = random.randint(0, 3) * 90  # 旋转
    hr_patch = hr_patch.rotate(angle, Image.BICUBIC, False, None)
    lr_patch = lr_patch.rotate(angle, Image.BICUBIC, False, None)

    return hr_patch, lr_patch


def resize(img: Image, size, interpolation=Image.BICUBIC):
    """
    Resize the input PIL Image to the given size.
    :param img: PIL.Image
    :param size: (h, w)
    :param interpolation: Image.BICUBIC
    :return: PIL.Image
    """
    return img.resize(size, interpolation)


def Random_pre_process_pair(lr: torch.Tensor, hr: torch.Tensor):
    latitude = torch.rand(1)
    longitude = torch.rand(1)
    rotate = torch.randint(0, 4, (1,)) * 90

    if latitude > 0.5:
        lr = torch.flip(lr, dims=[2])
        hr = torch.flip(hr, dims=[2])
    if longitude > 0.5:
        lr = torch.flip(lr, dims=[1])
        hr = torch.flip(hr, dims=[1])
    lr = torch.rot90(lr, int(rotate), dims=[1, 2])
    hr = torch.rot90(hr, int(rotate), dims=[1, 2])
    return lr, hr
