"""Mô phỏng nhiễu: AWGN (có thể mở rộng thêm Poisson, salt & pepper)."""
import numpy as np


def add_awgn(img, sigma, seed=None):
    """img float trong [0, 1], sigma tính trên thang [0, 1] (ví dụ 25/255)."""
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0.0, 1.0)
