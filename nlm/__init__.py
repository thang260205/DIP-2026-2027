"""Package Non-Local Means denoising."""
from .naive import nlm_naive
from .vectorized import nlm_vectorized
from .jit import nlm_jit


def nlm_denoise(img, f=2, t=5, h=0.1, sigma=None, backend="jit"):
    """img: float64 trong [0, 1]. backend: 'naive' | 'vectorized' | 'jit'."""
    fn = {"naive": nlm_naive, "vectorized": nlm_vectorized, "jit": nlm_jit}[backend]
    return fn(img, f=f, t=t, h=h, sigma=sigma)
