"""Kiểm thử tính đúng đắn của thuật toán NLM trên các backend:
1. Ảnh phẳng hằng số (flat image) không đổi sau khi lọc.
2. Shape đúng và giá trị kết quả nằm trong khoảng [0, 1].
3. PSNR và SSIM sau khi lọc cao hơn so với ảnh bị nhiễu (seed cố định).
"""
import pytest
import numpy as np
from skimage.data import camera

from nlm import nlm_denoise
from nlm.utils import to_float
from nlm.noise import add_awgn
from nlm.metrics import psnr, ssim


@pytest.mark.parametrize("backend", ["naive", "vectorized", "jit"])
def test_constant_flat_image(backend):
    """Ảnh phẳng hằng số không thay đổi sau khi lọc."""
    val = 0.6
    flat_img = np.full((24, 24), val, dtype=np.float64)
    filtered = nlm_denoise(flat_img, f=1, t=3, h=0.1, backend=backend)

    assert filtered.shape == flat_img.shape
    assert filtered.dtype == np.float64
    assert np.allclose(filtered, flat_img, atol=1e-6)


@pytest.mark.parametrize("backend", ["naive", "vectorized", "jit"])
def test_shape_and_range(backend):
    """Shape trả về đúng và mọi giá trị nằm nghiêm ngặt trong [0, 1]."""
    raw_img = to_float(camera()[:32, :32])
    filtered = nlm_denoise(raw_img, f=2, t=3, h=0.1, backend=backend)

    assert filtered.shape == raw_img.shape
    assert filtered.dtype == np.float64
    assert np.all(filtered >= 0.0)
    assert np.all(filtered <= 1.0)


@pytest.mark.parametrize("backend", ["naive", "vectorized", "jit"])
def test_psnr_and_ssim_improvement(backend):
    """PSNR và SSIM sau khi lọc bằng NLM phải cao hơn ảnh bị nhiễu."""
    clean = to_float(camera()[:36, :36])
    sigma = 25.0 / 255.0
    noisy = add_awgn(clean, sigma=sigma, seed=42)

    psnr_noisy = psnr(clean, noisy)
    ssim_noisy = ssim(clean, noisy)

    denoised = nlm_denoise(noisy, f=2, t=4, h=0.12, sigma=sigma, backend=backend)

    psnr_denoised = psnr(clean, denoised)
    ssim_denoised = ssim(clean, denoised)

    assert psnr_denoised > psnr_noisy
    assert ssim_denoised > ssim_noisy
