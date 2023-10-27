from torchvision import transforms
from PIL import Image
import torch

from utils.Prepro import random_pre_process_pair, Random_pre_process_pair


def defaultTransforms():
    return transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.ToTensor()
    ])


def ToTensor():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def resize(h, w):
    return transforms.Compose([
        transforms.Resize((h, w))
    ])


def openImg(filename, model='RGB'):
    return Image.open(filename).convert(model)


def tf(h, w, scale):
    return [transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((h, w)),
        transforms.ToTensor()
    ]), transforms.Compose([
        lambda x: Image.open(x).convert('RGB'),
        transforms.Resize((h * scale, w * scale)),
        transforms.ToTensor()
    ])]


def forward_chop(model, x, scale, shave=10, min_size=60000):
    n_GPUs = 1  # min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale, shave=shave, min_size=min_size)
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def div2k(hr, lr, patch_size, scale):
    hr = Image.open(hr).convert('RGB')
    lr = Image.open(lr).convert('RGB')
    return random_pre_process_pair(hr, lr, patch_size, scale)


def match_image(y: torch.Tensor, out: torch.Tensor):
    _, _, h1, w1 = y.shape
    _, _, h2, w2 = out.shape
    t = transforms.Resize((h1, w1))
    return t(out)


def RGB2YCbCr(rgb_image: torch.Tensor) -> torch.Tensor:
    rgb_image1 = rgb_image.clone()
    n_dim = rgb_image1.dim()
    transform_matrix = torch.tensor([[0.257, 0.504, 0.098],
                                     [-0.148, -0.291, 0.439],
                                     [0.439, -0.368, -0.071]], device=rgb_image1.device, dtype=rgb_image1.dtype)
    shift_matrix = torch.tensor([16, 128, 128], dtype=rgb_image1.dtype, device=rgb_image1.device)
    if n_dim == 4:  # [b c h w]
        b, c, h, w = rgb_image1.shape
        assert c == 3, 'Input image is not a RGB image!'
        YCbCr_image = torch.zeros_like(rgb_image1)
        for i in range(h):
            for j in range(w):
                YCbCr_image[:, :, i, j] = (transform_matrix @ rgb_image1[:, :, i, j].T).squeeze() + shift_matrix
        return YCbCr_image
    elif n_dim == 3:
        c, h, w = rgb_image1.shape
        assert c == 3, 'Input image is not a RGB image!'
        YCbCr_image = torch.zeros_like(rgb_image1)
        for i in range(h):
            for j in range(w):
                YCbCr_image[:, i, j] = transform_matrix @ rgb_image1[:, i, j] + shift_matrix
        return YCbCr_image
    else:
        raise ValueError('Do not support this shape input!')


def YCbCr2RGB():
    pass


def patch(image_lr: torch.Tensor, image_hr: torch.Tensor, scale=4, patch_size=48, batch_size=16):
    b, c, h, w = image_lr.shape
    assert b == 1, f'input batch_size must be 1 not {b}!'
    assert patch_size <= h or patch_size <= w, f'patch_size {patch_size} is out of h={h} w={w}!'
    images_lr = torch.zeros(batch_size, c, patch_size, patch_size, device=image_lr.device, dtype=image_lr.dtype)
    images_hr = torch.zeros(batch_size, c, patch_size * scale, patch_size * scale, device=image_hr.device,
                            dtype=image_hr.dtype)
    rand_h = torch.randint(0, h - patch_size, (batch_size,))
    rand_w = torch.randint(0, w - patch_size, (batch_size,))
    for i in range(batch_size):
        lr = image_lr[0, :, rand_h[i]:rand_h[i] + patch_size, rand_w[i]:rand_w[i] + patch_size]
        hr = image_hr[0, :, rand_h[i] * scale:(rand_h[i] + patch_size) * scale,
             rand_w[i] * scale:(rand_w[i] + patch_size) * scale]

        # 旋转和翻转
        lr, hr = Random_pre_process_pair(lr, hr)

        images_lr[i, ...] = lr
        images_hr[i, ...] = hr
    return images_lr, images_hr


if __name__ == '__main__':
    # x = torch.randn(1, 3, 252, 128)
    # y = torch.randn(1, 3, 250, 128)
    #
    # x = match_image(y, x)
    # print(x.shape)
    x = torch.randn(1, 3, 100, 100)
    y = torch.randn(1, 3, 400, 400)
    x1, y1 = patch(x, y)
    print(x1.shape)
