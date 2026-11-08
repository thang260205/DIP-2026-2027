"""Tiện ích: chuyển uint8 <-> float [0, 1], padding."""
import numpy as np


def to_float(img_uint8):
    return img_uint8.astype(np.float64) / 255.0


def to_uint8(img_float):
    return np.clip(np.round(img_float * 255.0), 0, 255).astype(np.uint8)


def pad_reflect(img, pad):
    return np.pad(img, pad, mode="reflect")
