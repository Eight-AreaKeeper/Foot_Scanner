"""合成测试图：用针孔相机模型渲染"脚踩在 A4 纸上"的照片。

脚轮廓上的每个点都带真实离地高度，所以能检验透视和视差校正是否正确。
它不模拟阴影、皮肤纹理和小腿遮挡，只用于验证几何流程，不能代替真实照片的精度测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .measure import DEFAULT_EDGE_HEIGHT_PROFILE
from .paper import PAPER_SIZES_MM


def _catmull_rom_closed(pts: np.ndarray, samples_per_seg: int = 24) -> np.ndarray:
    n = len(pts)
    out = []
    t = np.linspace(0, 1, samples_per_seg, endpoint=False)[:, None]
    for i in range(n):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    return np.vstack(out)


def foot_outline(length: float = 260.0, ball_width: float = 100.0, heel_width: float = 66.0) -> np.ndarray:
    """右脚俯视轮廓，局部坐标：脚跟最后点在原点，+x 指向脚尖，+y 为内侧。"""
    L, B, H = length, ball_width, heel_width
    ctrl = np.array(
        [
            (0.00, 0.00),
            (0.03, 0.40 * H),
            (0.16, 0.50 * H),
            (0.38, 0.40 * B),
            (0.70, 0.50 * B),
            (0.88, 0.42 * B),
            (0.965, 0.26 * B),
            (1.00, 0.10 * B),
            (0.975, -0.06 * B),
            (0.93, -0.24 * B),
            (0.84, -0.44 * B),
            (0.66, -0.50 * B),
            (0.40, -0.44 * B),
            (0.16, -0.50 * H),
            (0.03, -0.40 * H),
        ]
    )
    ctrl[:, 0] *= L
    return _catmull_rom_closed(ctrl)


@dataclass
class SyntheticScene:
    image: np.ndarray
    outline_mm: np.ndarray  # 脚轮廓在纸面坐标中的真实水平位置
    heights_mm: np.ndarray
    camera_center_mm: np.ndarray
    equiv_focal_mm: float


def _look_at(center: np.ndarray, target: np.ndarray, roll_deg: float) -> np.ndarray:
    fwd = target - center
    fwd /= np.linalg.norm(fwd)
    # 纸面坐标系：x 向右、y 向"下"（与图像一致），z 向上；相机向下看
    up_hint = np.array([0.0, -1.0, 0.0]) if abs(fwd[1]) < 0.95 else np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    r = np.vstack([right, down, fwd])  # 世界 -> 相机
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return roll @ r


def render(
    camera_center_mm=(105.0, 380.0, 600.0),
    target_mm=(105.0, 150.0, 0.0),
    roll_deg: float = 0.0,
    image_size=(3000, 4000),
    equiv_focal_mm: float = 26.0,
    paper: str = "a4",
    mode: str = "free",
    foot_kwargs: dict | None = None,
    foot_angle_deg: float = 0.0,
    foot_offset_mm=(0.0, 0.0),
    edge_height_profile=DEFAULT_EDGE_HEIGHT_PROFILE,
    seed: int = 0,
) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    w, h = image_size
    pw, ph = PAPER_SIZES_MM[paper]
    f = equiv_focal_mm / 43.27 * np.hypot(w, h)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    c = np.asarray(camera_center_mm, dtype=np.float64)
    # 渲染时 z 向上，而纸面坐标 (x 右, y 下) 是左手系，翻转 z 使其成为右手系再算位姿
    flip = np.diag([1.0, 1.0, -1.0])
    r = _look_at(flip @ c, flip @ np.asarray(target_mm, dtype=np.float64), roll_deg) @ flip

    def project(pts3):
        cam = (pts3 - c) @ r.T
        uv = cam @ k.T
        return uv[:, :2] / uv[:, 2:3]

    # 背景：深色带噪声的地面
    img = np.full((h, w, 3), (70, 75, 80), np.float32)
    img += rng.normal(0, 6, (h, w, 1))
    img = cv2.GaussianBlur(img, (0, 0), 1.0)

    paper_pts = np.array([[0, 0, 0], [pw, 0, 0], [pw, ph, 0], [0, ph, 0]], dtype=np.float64)
    paper_uv = project(paper_pts)
    layer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(layer, [np.round(paper_uv * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    alpha = layer.astype(np.float32)[:, :, None] / 255
    gy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    paper_color = np.array([238, 240, 242], np.float32) - 18 * gy  # 轻微光照梯度
    img = img * (1 - alpha) + paper_color * alpha

    local = foot_outline(**(foot_kwargs or {}))
    length = local[:, 0].max()
    t = np.clip(local[:, 0] / length, 0, 1)
    knots_t, knots_z = zip(*edge_height_profile)
    z = np.interp(t, knots_t, knots_z)

    ang = np.radians(foot_angle_deg)
    if mode == "wall":
        axis = np.array([0.0, 1.0])
        origin = np.array([pw / 2, 0.0]) + np.asarray(foot_offset_mm)
    else:
        axis = np.array([np.sin(ang), -np.cos(ang)])  # 默认脚尖朝纸的上边
        center = np.array([pw / 2, ph / 2]) + np.asarray(foot_offset_mm)
        origin = center - axis * length / 2
    normal = np.array([-axis[1], axis[0]])
    outline = origin + local[:, :1] * axis + local[:, 1:2] * normal

    foot_uv = project(np.column_stack([outline, z]))
    layer = np.zeros((h, w), np.uint8)
    cv2.fillPoly(layer, [np.round(foot_uv * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    alpha = layer.astype(np.float32)[:, :, None] / 255
    skin = np.array([135, 160, 205], np.float32) + rng.normal(0, 5, (h, w, 1)).astype(np.float32)
    img = img * (1 - alpha) + skin * alpha

    img += rng.normal(0, 2.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return SyntheticScene(
        image=img, outline_mm=outline, heights_mm=z, camera_center_mm=c, equiv_focal_mm=equiv_focal_mm
    )
