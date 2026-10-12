"""从纸面坐标（毫米）下的脚轮廓计算测量值。

坐标约定：先求脚轴（脚跟 -> 脚尖），再把轮廓旋转到"脚轴为 +x、脚跟在 x=0"
的局部坐标系里量长宽。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RASTER_PX_PER_MM = 5.0
BALL_REGION = (0.55, 0.85)  # 跖围宽在脚长 55%–85% 区间内取最大宽度
HEEL_POSITION = 0.16  # 脚跟宽取距脚跟 16% 脚长处
HEEL_BAND = 0.02

# 轮廓各处离地高度的经验值（毫米），按距脚跟的比例插值，用于视差校正。
# 脚跟最后点在跟骨后凸处，约 25 mm 高；前掌两侧和脚尖最外点约 10–12 mm。
# 这些值需要用真值数据标定。
DEFAULT_EDGE_HEIGHT_PROFILE = ((0.0, 25.0), (0.2, 12.0), (1.0, 10.0))


@dataclass
class FootFrame:
    origin: np.ndarray  # 脚跟最后点在纸面坐标中的位置（沿脚轴）
    axis: np.ndarray  # 单位向量，脚跟 -> 脚尖
    normal: np.ndarray  # 单位向量，axis 逆时针旋转 90°

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        rel = np.asarray(pts, dtype=np.float64) - self.origin
        return np.column_stack([rel @ self.axis, rel @ self.normal])

    def to_paper(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=np.float64)
        return self.origin + local[:, :1] * self.axis + local[:, 1:2] * self.normal


@dataclass
class Measurements:
    length_mm: float
    ball_width_mm: float
    ball_position: float  # 跖围宽所在位置（占脚长比例）
    heel_width_mm: float
    frame: FootFrame
    widths: np.ndarray  # (M, 2)：每个截面的 (x_mm, 宽度_mm)


def _slice_widths(local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把局部坐标下的轮廓栅格化，返回每一列的 x 位置和该处宽度。"""
    s = RASTER_PX_PER_MM
    xmin, ymin = local.min(axis=0) - 2
    xmax, ymax = local.max(axis=0) + 2
    w, h = int(np.ceil((xmax - xmin) * s)), int(np.ceil((ymax - ymin) * s))
    img = np.zeros((h, w), np.uint8)
    poly = np.round((local - [xmin, ymin]) * s * 16).astype(np.int32)  # 4 位小数精度
    cv2.fillPoly(img, [poly], 255, lineType=cv2.LINE_8, shift=4)
    cols = img > 0
    counts = cols.sum(axis=0)
    xs = xmin + (np.arange(w) + 0.5) / s
    has = counts > 0
    top = np.where(has, cols.argmax(axis=0), 0)
    bottom = np.where(has, h - 1 - cols[::-1].argmax(axis=0), 0)
    widths = np.where(has, (bottom - top + 1) / s, 0.0)
    return xs, widths


def _principal_axis(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def find_frame(contour_mm: np.ndarray, axis: np.ndarray | None = None, heel_x: float | None = None) -> FootFrame:
    """确定脚轴方向和脚跟位置。

    axis 为 None 时用主成分分析求脚轴，并用"脚跟比前掌窄"判断哪头是脚跟。
    heel_x 给定时（靠墙模式），脚跟位置固定在沿 axis 的该坐标处。
    """
    pts = np.asarray(contour_mm, dtype=np.float64)
    auto = axis is None
    a = _principal_axis(pts) if auto else np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    n = np.array([-a[1], a[0]])
    proj = pts @ a
    if auto:
        lo, hi = proj.min(), proj.max()
        frame = FootFrame(origin=lo * a, axis=a, normal=n)
        xs, widths = _slice_widths(frame.to_local(pts))
        length = hi - lo
        near_lo = widths[(xs > 0.1 * length) & (xs < 0.25 * length)].mean()
        near_hi = widths[(xs > 0.75 * length) & (xs < 0.9 * length)].mean()
        if near_lo > near_hi:  # 宽的一头是前掌，翻转使 lo 端为脚跟
            a, n = -a, -n
            proj = -proj
    start = proj.min() if heel_x is None else heel_x
    # 原点放在脚跟处、横向放在脚轴上，使局部 y 以脚轴为中心
    center_offset = (pts @ n).mean()
    origin = start * a + center_offset * n
    return FootFrame(origin=origin, axis=a, normal=n)


def edge_heights(contour_mm: np.ndarray, frame: FootFrame, length_mm: float, profile=DEFAULT_EDGE_HEIGHT_PROFILE) -> np.ndarray:
    t = np.clip(frame.to_local(contour_mm)[:, 0] / max(length_mm, 1e-6), 0, 1)
    knots_t, knots_z = zip(*profile)
    return np.interp(t, knots_t, knots_z)


def measure(contour_mm: np.ndarray, frame: FootFrame) -> Measurements:
    local = frame.to_local(contour_mm)
    length = float(local[:, 0].max())
    xs, widths = _slice_widths(local)

    in_ball = (xs >= BALL_REGION[0] * length) & (xs <= BALL_REGION[1] * length)
    i = int(np.argmax(np.where(in_ball, widths, -1)))
    ball_width, ball_pos = float(widths[i]), float(xs[i] / length)

    in_heel = np.abs(xs / length - HEEL_POSITION) <= HEEL_BAND
    heel_width = float(widths[in_heel].mean())

    keep = widths > 0
    return Measurements(
        length_mm=length,
        ball_width_mm=ball_width,
        ball_position=ball_pos,
        heel_width_mm=heel_width,
        frame=frame,
        widths=np.column_stack([xs[keep], widths[keep]]),
    )
