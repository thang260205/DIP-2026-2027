"""Các chỉ số đánh giá: MSE, PSNR, SSIM."""
import numpy as np
from skimage.metrics import structural_similarity


def mse(ref, img):
    return float(np.mean((ref - img) ** 2))


def psnr(ref, img, max_i=1.0):
    m = mse(ref, img)
    return float("inf") if m == 0 else 10 * np.log10(max_i ** 2 / m)


def ssim(ref, img, data_range=1.0):
    return float(structural_similarity(ref, img, data_range=data_range))
