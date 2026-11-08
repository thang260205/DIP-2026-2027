"""Thuật toán Non-Local Means (NLM) phiên bản JIT biên dịch đa luồng bằng Numba.

Khai thác Numba @njit(parallel=True) với prange trên vòng lặp các hàng của ảnh:
- Giữ nguyên độ chính xác giải tích của bản chuẩn naive (sai khác < 1e-6).
- Tăng tốc tối đa nhờ biên dịch mã máy song song hóa đa luồng trên CPU.
"""
import numpy as np
from numba import njit, prange
from .utils import pad_reflect


@njit(parallel=True)
def _nlm_jit_core(pad_img, img, f, t, h2):
    """Hàm lõi tính toán NLM biên dịch JIT với Numba, song song hóa theo hàng."""
    m, n = img.shape
    out = np.empty((m, n), dtype=np.float64)
    pad = t + f
    patch_size = (2 * f + 1) * (2 * f + 1)

    # Song song hóa qua từng hàng của ảnh với prange
    for i in prange(m):
        for j in range(n):
            r = i + pad
            c = j + pad

            max_w = 0.0
            sum_w = 0.0
            sum_val = 0.0

            # Lặp qua tất cả điểm q trong vùng tìm kiếm Omega_p
            for di in range(-t, t + 1):
                for dj in range(-t, t + 1):
                    # Bỏ qua điểm tự thân q = p trong vòng lặp lân cận
                    if di == 0 and dj == 0:
                        continue

                    # Bước 2: Tính khoảng cách patch d²(p, q) theo mục 4.2.1
                    diff_sum = 0.0
                    for dy in range(-f, f + 1):
                        for dx in range(-f, f + 1):
                            diff = (
                                pad_img[r + dy, c + dx]
                                - pad_img[r + di + dy, c + dj + dx]
                            )
                            diff_sum += diff * diff

                    d2 = diff_sum / patch_size

                    # Bước 3: Tính trọng số hàm mũ w(p, q) theo mục 4.2.2
                    w = np.exp(-d2 / h2)
                    if w > max_w:
                        max_w = w
                    sum_w += w
                    sum_val += w * pad_img[r + di, c + dj]

            # Bước 4: Sửa kỳ dị tự trọng số w(p, p) = max_{q != p} w(p, q) theo mục 4.2.3
            sum_w += max_w
            sum_val += max_w * pad_img[r, c]

            # Bước 5: Chuẩn hóa Z(p) và tính u_hat(p) theo mục 4.2
            if sum_w > 0.0:
                out[i, j] = sum_val / sum_w
            else:
                out[i, j] = pad_img[r, c]

    return out


def nlm_jit(img, f=2, t=5, h=0.1, sigma=None) -> np.ndarray:
    """Khử nhiễu ảnh bằng Non-Local Means phiên bản JIT Numba đa luồng.

    Tham số:
        img: np.ndarray float64 trong đoạn [0, 1], kích thước (M, N).
        f: int, bán kính khối patch (kích thước patch là (2f + 1) x (2f + 1)).
        t: int, bán kính vùng tìm kiếm (kích thước cửa sổ là (2t + 1) x (2t + 1)).
        h: float, tham số lọc (trên thang [0, 1]).
        sigma: float hoặc None, độ lệch chuẩn nhiễu (tham khảo).

    Trả về:
        np.ndarray float64 cùng kích thước với img, giá trị trong đoạn [0, 1].
    """
    img = np.asarray(img, dtype=np.float64)
    h2 = float(h * h)

    # Bước 1: Mở rộng biên ảnh (padding) bằng phản xạ đối xứng theo mục 4.2
    pad = t + f
    pad_img = pad_reflect(img, pad)

    # Gọi hàm lõi Numba đa luồng
    out = _nlm_jit_core(pad_img, img, f, t, h2)

    # Đảm bảo miền giá trị kết quả thuộc [0, 1]
    return np.clip(out, 0.0, 1.0)
