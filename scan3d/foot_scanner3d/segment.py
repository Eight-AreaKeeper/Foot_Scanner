"""脚部分割（原型：颜色阈值）。

真实场景要换成学习型分割（iOS Vision 人像分割或自训的轻量 UNet）；这里只假设
脚是画面里饱和度明显高于白纸和灰色地面的连通区域。
"""

from __future__ import annotations

import cv2
import numpy as np


def segment_foot(image: np.ndarray, min_saturation: int = 45, min_value: int = 45) -> np.ndarray:
    """返回 uint8 掩码（255 = 脚）。"""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    skin = (sat >= min_saturation) & (val >= min_value) & ((hue <= 25) | (hue >= 165))
    mask = skin.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == keep).astype(np.uint8) * 255
    # 填掉内部小洞（脚趾高光等）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def signed_distance(mask: np.ndarray, pad: int = 64) -> np.ndarray:
    """轮廓有符号距离（像素，内部为正）。

    先用边缘像素外扩，避免脚伸出画面时把画面边界当成轮廓。
    """
    m = cv2.copyMakeBorder((mask > 0).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sd = np.where(m > 0, d_in - 0.5, -(d_out - 0.5)).astype(np.float32)
    return sd[pad:-pad, pad:-pad]
