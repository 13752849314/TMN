# create by 敖鸥 at 2023/6/6
from utils.data import RGB2YCbCr
from . import psnr, ssim


def evaluation(out, y, isY=True):
    dim1 = out.dim()
    dim2 = y.dim()
    assert dim1 == dim2 and dim1 in [3, 4], f'dim1 {dim1} != dim2 {dim2}'
    out1 = RGB2YCbCr(out)
    y1 = RGB2YCbCr(y)

    if not isY:
        return psnr(out1, y1), ssim(out, y1)
    else:
        if dim1 == 4:
            return psnr(out1[:, 0, ...], y1[:, 0, ...]), ssim(out1[:, 0, ...], y1[:, 0, ...])
        else:
            return psnr(out1[0, ...], y1[0, ...]), ssim(out1[0, ...], y1[0, ...])
