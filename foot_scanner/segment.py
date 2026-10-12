"""在校正后的纸面俯视图中分割脚的轮廓。

纸是近似均匀的白色，脚（皮肤或袜子）要么颜色更饱和、要么更暗。先用纸的
边缘一圈拟合出纸面的亮度分布（应对光照不均），再按"比纸暗多少 + 偏离纸色
多少"打分，用 Otsu 自动取阈值。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# 比纸暗 + 2×偏色 至少到这个分数才算脚；皮肤约 100+，浅灰阴影约 10–30
MIN_SCORE = 35


class FootNotFound(RuntimeError):
    pass


@dataclass
class FootMask:
    mask: np.ndarray  # uint8，255 为脚
    contour_px: np.ndarray  # (N, 2) 俯视图像素坐标
    touches_edge: dict[str, bool]  # 纸的四条边：top/bottom/left/right


def _fit_background(lum: np.ndarray, band: np.ndarray) -> np.ndarray:
    """用纸边一圈的像素拟合二次曲面，作为整张纸的亮度背景。"""
    h, w = lum.shape
    ys, xs = np.nonzero(band)
    vals = lum[ys, xs].astype(np.float64)
    xn, yn = xs / w - 0.5, ys / h - 0.5
    a = np.column_stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn])
    keep = np.ones(len(vals), bool)
    for _ in range(3):  # 迭代剔除落在脚上或阴影里的点
        coef, *_ = np.linalg.lstsq(a[keep], vals[keep], rcond=None)
        resid = vals - a @ coef
        sigma = 1.4826 * np.median(np.abs(resid[keep])) + 1e-6
        keep = resid > -3 * sigma
    gy, gx = np.mgrid[0:h, 0:w]
    gxn, gyn = gx / w - 0.5, gy / h - 0.5
    return coef[0] + coef[1] * gxn + coef[2] * gyn + coef[3] * gxn**2 + coef[4] * gxn * gyn + coef[5] * gyn**2


def segment_foot(warped: np.ndarray, px_per_mm: float) -> FootMask:
    h, w = warped.shape[:2]
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum, a, b = lab[:, :, 0], lab[:, :, 1] - 128, lab[:, :, 2] - 128

    margin = max(2, int(round(1.5 * px_per_mm)))  # 纸边 1.5 mm 不参与（角点误差会带进地面）
    band_w = max(margin + 2, int(round(12 * px_per_mm)))
    band = np.zeros((h, w), bool)
    band[margin:band_w, margin:-margin] = True
    band[-band_w:-margin, margin:-margin] = True
    band[margin:-margin, margin:band_w] = True
    band[margin:-margin, -band_w:-margin] = True

    bg = _fit_background(lum, band)
    a0, b0 = np.median(a[band]), np.median(b[band])
    darker = np.clip(bg - lum, 0, None)
    chroma = np.hypot(a - a0, b - b0)
    score = darker + 2.0 * chroma
    score = cv2.GaussianBlur(score, (0, 0), 0.5 * px_per_mm)
    score[:margin, :] = score[-margin:, :] = 0
    score[:, :margin] = score[:, -margin:] = 0

    # 分数用绝对刻度（OpenCV 8 位 Lab），不做归一化，否则空白纸上的噪声会被放大成"脚"
    s8 = np.clip(score, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(thr, MIN_SCORE)
    mask = (s8 > thr).astype(np.uint8) * 255

    k = max(3, int(round(2 * px_per_mm)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise FootNotFound("纸上没有找到脚")
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area_mm2 = stats[idx, cv2.CC_STAT_AREA] / px_per_mm**2
    if area_mm2 < 5000:  # 成人脚印面积远大于 50 cm²
        raise FootNotFound("纸上的深色区域太小，不像一只脚")
    mask = (labels == idx).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)  # 填掉内部空洞

    tol = margin + max(1, int(round(0.5 * px_per_mm)))
    x, y, bw, bh = cv2.boundingRect(contour)
    touches = {
        "top": y <= tol,
        "left": x <= tol,
        "bottom": y + bh >= h - tol,
        "right": x + bw >= w - tol,
    }
    return FootMask(mask=mask, contour_px=contour.reshape(-1, 2).astype(np.float64), touches_edge=touches)
