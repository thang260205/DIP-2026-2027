"""Đo đạc và đánh giá hiệu năng (thời gian thực thi) của các backend NLM và thư viện scikit-image.

Quy trình thực hiện:
- Kiểm thử trên cùng một ảnh gốc ở hai kích thước: 128x128 và 256x256.
- Đo thời gian cho 4 backend: naive, vectorized, jit, skimage.
- Khởi động (warmup) backend jit trước khi đo để loại trừ chi phí biên dịch JIT.
- Lặp lại nhiều lần (3 - 10 lần) để tính giá trị trung bình (time_mean) và độ lệch chuẩn (time_std).
- Ghi kết quả vào file results/runtime.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from skimage.data import camera
from skimage.restoration import denoise_nl_means

from nlm.naive import nlm_naive
from nlm.vectorized import nlm_vectorized
from nlm.jit import nlm_jit
from nlm.utils import to_float

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_benchmark():
    """Thực thi toàn bộ quy trình đo kiểm hiệu năng và lưu kết quả vào CSV."""
    f = 2
    t = 5
    h = 0.1
    sizes = [128, 256]

    # Lặp 3 lần cho Naive và 5 - 10 lần cho các backend khác
    repeats_config = {
        128: {"naive": 3, "vectorized": 5, "jit": 10, "skimage": 5},
        256: {"naive": 3, "vectorized": 5, "jit": 10, "skimage": 5},
    }

    full_cam = to_float(camera())
    records = []

    # Khởi động (warmup) Numba JIT để biên dịch trước khi bắt đầu đo
    print("[Warmup] Dang khoi dong va bien dich Numba JIT...")
    dummy = full_cam[:32, :32]
    nlm_jit(dummy, f=f, t=t, h=h)
    print("[Warmup] Hoan tat bien dich JIT.\n")

    for size in sizes:
        print(f"=== Bat dau do hieu nang tren anh {size}x{size} ===")
        img = full_cam[:size, :size]

        backends = [
            ("naive", lambda im: nlm_naive(im, f=f, t=t, h=h)),
            ("vectorized", lambda im: nlm_vectorized(im, f=f, t=t, h=h)),
            ("jit", lambda im: nlm_jit(im, f=f, t=t, h=h)),
            (
                "skimage",
                lambda im: denoise_nl_means(
                    im,
                    patch_size=2 * f + 1,
                    patch_distance=t,
                    h=h,
                    fast_mode=False,
                ),
            ),
        ]

        for name, fn in backends:
            n_reps = repeats_config[size][name]
            print(f"  -> Do backend '{name}' ({n_reps} lan lap)...", end="", flush=True)

            durations = []
            for rep in range(n_reps):
                t0 = time.perf_counter()
                fn(img)
                t1 = time.perf_counter()
                durations.append(t1 - t0)

            t_mean = float(np.mean(durations))
            t_std = float(np.std(durations)) if len(durations) > 1 else 0.0

            print(f" Xong! Mean: {t_mean:.4f}s | Std: {t_std:.4f}s")

            records.append({
                "backend": name,
                "size": f"{size}x{size}",
                "time_mean": t_mean,
                "time_std": t_std,
            })

    # Tạo thư mục results nếu chưa tồn tại
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "runtime.csv")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"\n[Ket qua] Da luu bang hieu nang vao: {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
