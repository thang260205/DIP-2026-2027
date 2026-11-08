"""Thuật toán Non-Local Means (NLM) phiên bản vòng lặp thuần (naive).

Cài đặt bám sát từng bước công thức giải tích trong Báo cáo Chương 4:
- Khoảng cách patch theo mục 4.2.1
- Trọng số hàm mũ theo mục 4.2.2
- Khắc phục điểm kỳ dị tự trọng số theo mục 4.2.3
- Chuẩn hóa và ước lượng u_hat(p) theo mục 4.2
"""
import numpy as np
from .utils import pad_reflect


def nlm_naive(img, f=2, t=5, h=0.1, sigma=None) -> np.ndarray:
    """Khử nhiễu ảnh bằng Non-Local Means phiên bản vòng lặp thuần (chuẩn đối chiếu).

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
    patch_size = (2 * f + 1) ** 2

    # Bước 1: Mở rộng biên ảnh (padding) bằng phản xạ đối xứng theo mục 4.2
    pad = t + f
    pad_img = pad_reflect(img, pad)

    out = np.zeros((m, n), dtype=np.float64)

    # Lặp qua từng điểm ảnh p = (i, j) trong lưới Omega
    for i in range(m):
        for j in range(n):
            r = i + pad
            c = j + pad
            patch_p = pad_img[r - f : r + f + 1, c - f : c + f + 1]

            max_w = 0.0
            sum_w = 0.0
            sum_val = 0.0

            # Lặp qua tất cả điểm q trong vùng tìm kiếm Omega_p
            for di in range(-t, t + 1):
                for dj in range(-t, t + 1):
                    # Bỏ qua điểm tự thân q = p trong vòng lặp tính trọng số lân cận
                    if di == 0 and dj == 0:
                        continue

                    # Bước 2: Tính khoảng cách tương đồng patch d²(p, q) theo mục 4.2.1
                    patch_q = pad_img[
                        r + di - f : r + di + f + 1,
                        c + dj - f : c + dj + f + 1
                    ]
                    diff = patch_p - patch_q
                    d2 = np.sum(diff * diff) / patch_size

                    # Bước 3: Tính trọng số hàm mũ w(p, q) theo mục 4.2.2
                    w = np.exp(-d2 / h2)

                    if w > max_w:
                        max_w = w
                    sum_w += w
                    sum_val += w * pad_img[r + di, c + dj]

            # Bước 4: Sửa kỳ dị tự trọng số w(p, p) = max_{q != p} w(p, q) theo mục 4.2.3
            sum_w += max_w
            sum_val += max_w * pad_img[r, c]

            # Bước 5: Chuẩn hóa với Z(p) và tính u_hat(p) theo mục 4.2
            if sum_w > 0.0:
                out[i, j] = sum_val / sum_w
            else:
                out[i, j] = pad_img[r, c]

    # Đảm bảo miền giá trị kết quả thuộc [0, 1]
    return np.clip(out, 0.0, 1.0)
