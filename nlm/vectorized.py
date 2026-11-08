"""Thuật toán Non-Local Means (NLM) phiên bản vector hóa bằng NumPy và Integral Image.

Khai thác giải thuật của Darbon et al. (2008):
- Lặp theo từng vector dịch delta = (di, dj) trong vùng tìm kiếm Omega_p.
- Xây dựng mảng tích lũy 2D (Integral Image / Summed-Area Table) để tính khoảng
  cách patch d²(p, p + delta) cho toàn bộ điểm ảnh p cùng lúc với chi phí O(1).
- Hoàn toàn không sử dụng vòng lặp qua từng điểm ảnh (pixel loops).
"""
import numpy as np
from .utils import pad_reflect


def nlm_vectorized(img, f=2, t=5, h=0.1, sigma=None) -> np.ndarray:
    """Khử nhiễu ảnh bằng Non-Local Means phiên bản vector hóa (NumPy + Integral Image).

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
    m, n = img.shape
    h2 = h * h
    patch_len = 2 * f + 1
    patch_size = patch_len * patch_len

    # Bước 1: Mở rộng biên ảnh (padding) bằng phản xạ đối xứng theo mục 4.2
    pad = t + f
    pad_img = pad_reflect(img, pad)

    # Khởi tạo các mảng tích lũy cho toàn bộ ảnh (không lặp qua pixel)
    sum_w = np.zeros((m, n), dtype=np.float64)
    sum_val = np.zeros((m, n), dtype=np.float64)
    max_w = np.zeros((m, n), dtype=np.float64)

    # Phạm vi trích xuất vùng patch cho mọi pixel trong ảnh gốc
    r_start = t
    r_end = t + m + 2 * f
    c_start = t
    c_end = t + n + 2 * f
    v1 = pad_img[r_start:r_end, c_start:c_end]

    # Lặp qua từng vector dịch chuyển delta = (di, dj) trong cửa sổ tìm kiếm (Darbon et al., 2008)
    for di in range(-t, t + 1):
        for dj in range(-t, t + 1):
            # Bỏ qua vector dịch (0, 0) tương ứng với điểm tự thân q = p
            if di == 0 and dj == 0:
                continue

            # Bước 2: Tính ma trận sai số bình phương S_delta và Integral Image theo mục 4.4
            v2 = pad_img[r_start + di : r_end + di, c_start + dj : c_end + dj]
            diff = v1 - v2
            sq_diff = diff * diff

            # Xây dựng mảng tích lũy 2D (Summed-Area Table) với hàng và cột 0 đệm đầu
            sat = np.zeros((sq_diff.shape[0] + 1, sq_diff.shape[1] + 1), dtype=np.float64)
            sat[1:, 1:] = np.cumsum(np.cumsum(sq_diff, axis=0), axis=1)

            # Truy xuất tổng sai số khối (2f + 1) x (2f + 1) cho mọi pixel cùng lúc trong O(1)
            box_sum = (
                sat[patch_len : m + patch_len, patch_len : n + patch_len]
                - sat[0 : m, patch_len : n + patch_len]
                - sat[patch_len : m + patch_len, 0 : n]
                + sat[0 : m, 0 : n]
            )

            # d²(p, q) theo mục 4.2.1
            d2 = box_sum / patch_size

            # Bước 3: Tính ma trận trọng số w(p, q) theo mục 4.2.2
            w = np.exp(-d2 / h2)

            # Cập nhật trọng số cực đại w_max cho từng pixel
            max_w = np.maximum(max_w, w)
            sum_w += w

            # v(q) tương ứng với vị trí bị dịch chuyển (di, dj) trong ảnh đệm
            v_q = pad_img[pad + di : pad + m + di, pad + dj : pad + n + dj]
            sum_val += w * v_q

    # Bước 4: Sửa điểm kỳ dị tự trọng số w(p, p) = max_{q != p} w(p, q) theo mục 4.2.3
    sum_w += max_w
    sum_val += max_w * img

    # Bước 5: Chuẩn hóa bằng Z(p) và ước lượng u_hat(p) theo mục 4.2
    out = np.where(sum_w > 0.0, sum_val / sum_w, img)

    # Đảm bảo miền giá trị kết quả thuộc [0, 1]
    return np.clip(out, 0.0, 1.0)
