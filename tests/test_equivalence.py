"""Kiểm thử tính tương đương giữa các backend của NLM và đối sánh với skimage:
1. So sánh naive, vectorized, jit trên ảnh thực tế (camera 32x32), yêu cầu sai khác tối đa < 1e-6.
2. So sánh với skimage.restoration.denoise_nl_means, in sai khác trung bình tuyệt đối (MAE).
"""
import numpy as np
from skimage.data import camera
from skimage.restoration import denoise_nl_means

from nlm.naive import nlm_naive
from nlm.vectorized import nlm_vectorized
from nlm.jit import nlm_jit
from nlm.utils import to_float


def test_backends_equivalence():
    """Kiểm tra 3 backend (naive, vectorized, jit) cho kết quả đồng nhất (sai khác < 1e-6)."""
    img = to_float(camera()[:32, :32])
    f, t, h = 2, 5, 0.1

    out_naive = nlm_naive(img, f=f, t=t, h=h)
    out_vectorized = nlm_vectorized(img, f=f, t=t, h=h)
    out_jit = nlm_jit(img, f=f, t=t, h=h)

    diff_naive_vec = np.max(np.abs(out_naive - out_vectorized))
    diff_naive_jit = np.max(np.abs(out_naive - out_jit))
    diff_vec_jit = np.max(np.abs(out_vectorized - out_jit))

    print(f"\nSai khác tuyệt đối tối đa (naive vs vectorized): {diff_naive_vec:.2e}")
    print(f"Sai khác tuyệt đối tối đa (naive vs jit): {diff_naive_jit:.2e}")
    print(f"Sai khác tuyệt đối tối đa (vectorized vs jit): {diff_vec_jit:.2e}")

    assert diff_naive_vec < 1e-6, f"naive vs vectorized lệch {diff_naive_vec}"
    assert diff_naive_jit < 1e-6, f"naive vs jit lệch {diff_naive_jit}"
    assert diff_vec_jit < 1e-6, f"vectorized vs jit lệch {diff_vec_jit}"


def test_comparison_with_skimage():
    """Đối sánh với hàm chuẩn skimage.restoration.denoise_nl_means và in sai khác."""
    img = to_float(camera()[:32, :32])
    f, t, h = 2, 5, 0.1

    out_our = nlm_jit(img, f=f, t=t, h=h)
    out_skimage = denoise_nl_means(
        img,
        patch_size=2 * f + 1,
        patch_distance=t,
        h=h,
        fast_mode=False
    )

    mae = float(np.mean(np.abs(out_our - out_skimage)))
    max_diff = float(np.max(np.abs(out_our - out_skimage)))

    print(f"\n[Đối sánh scikit-image]")
    print(f"Sai khác trung bình tuyệt đối (MAE): {mae:.6f}")
    print(f"Sai khác tuyệt đối cực đại (Max Diff): {max_diff:.6f}")

    # Không đặt ngưỡng cứng cho skimage theo Quy tắc 7, chỉ kiểm tra tính hợp lệ số học
    assert np.all(np.isfinite(out_skimage))
    assert np.all(out_skimage >= 0.0)
    assert np.all(out_skimage <= 1.0)
