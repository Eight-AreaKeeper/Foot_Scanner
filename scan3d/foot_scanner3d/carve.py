"""轮廓雕刻（visual hull）重建。

每个体素投影到每一帧，取该帧脚轮廓的有符号距离（换算成毫米），
所有帧取最小值得到一个隐式场：>0 表示在所有视角下都落在脚的轮廓里。
用连续的距离场而不是 0/1 体素，可以让 marching cubes 得到亚体素精度的表面。

visual hull 的局限：只能恢复从侧面和上方"看得到轮廓"的形状。脚趾缝、
足弓底部这类从任何视角都被遮住的凹陷会被填平；鞋楦用的围度和宽度本来就是
皮尺/卡尺量出的外包络，受影响很小，但足弓高度不能从这里得到。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .mesh import Mesh, mesh_from_field

_OUTSIDE_IMAGE = 1e6  # 投影到画面外：该帧不提供约束


@dataclass
class View:
    camera: Camera
    sdf_px: np.ndarray  # 轮廓有符号距离（像素，内部为正）


def _sample_view(view: View, pts: np.ndarray) -> np.ndarray:
    cam = view.camera
    uv, z = cam.project(pts)
    n = len(pts)
    cols = 4096
    rows = -(-n // cols)
    mx = np.full(rows * cols, -1.0, np.float32)
    my = np.full(rows * cols, -1.0, np.float32)
    mx[:n], my[:n] = uv[:, 0], uv[:, 1]
    s = cv2.remap(
        view.sdf_px,
        mx.reshape(rows, cols),
        my.reshape(rows, cols),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_OUTSIDE_IMAGE,
    ).ravel()[:n]
    out = s * (z / cam.focal)
    out[(s >= _OUTSIDE_IMAGE * 0.5) | (z <= 0)] = np.inf
    return out.astype(np.float32)


def carve_field(views: list[View], lo: np.ndarray, hi: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """在 [lo, hi] 的体素中心上计算雕刻场。返回 (field, 第一个体素中心坐标)。"""
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    field = np.full(len(pts), np.inf, np.float32)
    for v in views:
        np.minimum(field, _sample_view(v, pts), out=field)
    field[~np.isfinite(field)] = -voxel  # 没有任何一帧看到的体素按外部处理
    origin = np.array([a[0] for a in axes])
    return field.reshape(gx.shape), origin


def _grid(lo, hi, voxel):
    axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
    g = np.meshgrid(*axes, indexing="ij")
    return np.stack([a.ravel() for a in g], axis=1)


def robust_hull_points(views: list[View], lo, hi, voxel: float = 4.0, outliers: int = 0) -> np.ndarray:
    """粗分辨率的稳健 visual hull：允许最多 outliers 帧"不同意"，返回被占据的体素中心。

    用于找出位姿错误或分割失败的帧：这些帧会把真实的脚也雕掉，
    取第 (outliers+1) 小的值就能把它们的影响排除在外。
    """
    pts = _grid(np.asarray(lo, float), np.asarray(hi, float), voxel)
    vals = np.stack([_sample_view(v, pts) for v in views])
    k = min(outliers, len(views) - 1)
    kth = np.partition(vals, k, axis=0)[k]
    return pts[kth > 0]


def outside_fraction(view: View, pts: np.ndarray, tol_mm: float = 10.0) -> float:
    """hull 体素中明显落在该帧轮廓外（超过 tol_mm）的比例。正确的帧应接近 0。"""
    d = _sample_view(view, pts)
    return float(np.mean(d < -tol_mm)) if len(pts) else 1.0


def carve(
    views: list[View],
    region_lo: np.ndarray,
    region_hi: np.ndarray,
    voxel: float = 1.0,
    coarse_voxel: float = 4.0,
    margin: float = 8.0,
) -> Mesh:
    """先粗后细：粗网格找到脚的包围盒，再在包围盒内细雕刻并提取网格。

    region 的底面应是纸面 z = 0；顶面截断（小腿以上不需要）。
    """
    region_lo, region_hi = np.asarray(region_lo, float), np.asarray(region_hi, float)
    coarse, c_origin = carve_field(views, region_lo, region_hi, coarse_voxel)
    occ = np.argwhere(coarse > 0)
    if len(occ) == 0:
        raise RuntimeError("雕刻结果为空：位姿或分割有问题")
    lo = np.maximum(c_origin + occ.min(axis=0) * coarse_voxel - margin, region_lo)
    hi = np.minimum(c_origin + occ.max(axis=0) * coarse_voxel + margin, region_hi)
    lo[2], hi[2] = region_lo[2], region_hi[2]

    field, origin = carve_field(views, lo, hi, voxel)
    # 上下各补一层镜像负值，使等值面恰好落在 z=lo 和 z=hi（地面和截断面）上
    field = np.concatenate([-np.abs(field[:, :, :1]), field, -np.abs(field[:, :, -1:])], axis=2)
    origin = origin - np.array([0, 0, voxel])
    # 四周补一圈外部值以封闭网格
    field = np.pad(field, ((1, 1), (1, 1), (0, 0)), constant_values=-voxel)
    origin = origin - np.array([voxel, voxel, 0])
    return mesh_from_field(field, origin, voxel)
