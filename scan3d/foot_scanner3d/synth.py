"""合成数据：参数化脚模型 + 手机绕脚拍摄的渲染。

脚模型由若干椭球和圆锥台平滑拼接而成（隐式曲面），包含脚跟、足弓、跖骨区、
五个分开的脚趾和一段小腿。它不追求解剖学精确，只要求：尺寸在成年人正常范围、
有真实脚会带来的难点（脚趾缝、内侧足弓凹陷、小腿遮挡纸面）。真值测量直接在
高精度真值网格上用同一套测量代码算出，所以误差只来自采集和重建。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera, intrinsics
from .mesh import Mesh, mesh_from_field

A4_MM = (210.0, 297.0)
SKIN_BGR = np.array([150.0, 178.0, 228.0])
PAPER_BGR = np.array([232.0, 234.0, 236.0])
# 四圈仰角：贴地一圈（约 15 cm 高）压住脚掌两侧下沿，近俯视一圈压住脚背顶部。
# 只拍两圈（30°/55°）时 visual hull 在脚背和跖围处会偏大 3–5 mm，见 README。
DEFAULT_ELEVATIONS = (15.0, 35.0, 60.0, 82.0)


@dataclass
class FootShape:
    length_scale: float = 1.0
    width_scale: float = 1.0
    height_scale: float = 1.0
    arch_depth_mm: float = 14.0
    leg_top_mm: float = 230.0

    @staticmethod
    def random(rng: np.random.Generator) -> "FootShape":
        return FootShape(
            length_scale=rng.uniform(0.9, 1.1),
            width_scale=rng.uniform(0.9, 1.12),
            height_scale=rng.uniform(0.9, 1.12),
            arch_depth_mm=rng.uniform(8.0, 18.0),
        )


# ---------- 隐式曲面工具（负值在内部） ----------


def _sd_ellipsoid(p, c, r):
    r = np.asarray(r, float)
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def _sd_round_cone(p, a, b, ra, rb):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba, pa = b - a, p - a
    h = np.clip(pa @ ba / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - (ra + (rb - ra) * h)


def _smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


# 脚坐标系：x 从脚跟指向脚趾，y 负方向为内侧（大脚趾一侧），z 朝上，单位 mm
_TOES = [  # (y, 根部半径, 趾尖半径, 趾尖 x)
    (-34.0, 15.0, 12.0, 258.0),
    (-12.0, 9.5, 8.0, 254.0),
    (5.0, 8.5, 7.5, 245.0),
    (20.0, 8.0, 7.0, 234.0),
    (34.0, 8.0, 7.0, 220.0),
]


def foot_sdf(p_mm: np.ndarray, shape: FootShape) -> np.ndarray:
    s = np.array([shape.length_scale, shape.width_scale, shape.height_scale])
    p = p_mm / s
    # 承重状态：脚底软组织被压平，椭球中心低于半径，被地面截出接触面
    body = _sd_ellipsoid(p, (38, 0, 26), (38, 31, 32))
    body = _smin(body, _sd_ellipsoid(p, (105, -2, 28), (55, 40, 34)), 22)
    body = _smin(body, _sd_ellipsoid(p, (178, -2, 15), (38, 51, 29)), 22)
    toes = None
    for y, r0, r1, tip in _TOES:
        d = _sd_round_cone(p, (192, y, 15), (tip - r1, y, r1), r0, r1)
        toes = d if toes is None else np.minimum(toes, d)
    body = _smin(body, toes, 7)
    leg = _sd_round_cone(p, (50, 0, 55), (40, 0, 400), 33, 36)
    body = _smin(body, leg, 16)
    d = body * s.min()
    arch = _sd_ellipsoid(p_mm, (100 * s[0], -40 * s[1], 0), (50 * s[0], 24 * s[1], shape.arch_depth_mm))
    d = _smax(d, -arch, 4)
    d = np.maximum(d, -p_mm[:, 2])  # 地面
    return np.maximum(d, p_mm[:, 2] - shape.leg_top_mm)


def foot_mesh(shape: FootShape, spacing: float = 0.8, z_max: float | None = None) -> Mesh:
    """在脚坐标系中提取脚模型网格。"""
    s = np.array([shape.length_scale, shape.width_scale, shape.height_scale])
    lo = np.array([-8.0 * s[0], -65.0 * s[1], -spacing])
    hi = np.array([268.0 * s[0], 65.0 * s[1], (z_max if z_max is not None else shape.leg_top_mm) + spacing])
    axes = [np.arange(lo[i], hi[i] + spacing, spacing) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    d = np.concatenate([foot_sdf(c, shape) for c in np.array_split(pts, max(1, len(pts) // 2_000_000))])
    field = -d.reshape(gx.shape).astype(np.float32)
    if z_max is not None:
        field[:, :, -1] = -np.abs(field[:, :, -1])  # 顶部封口
    return mesh_from_field(field, lo, spacing)


# ---------- 场景与渲染 ----------


@dataclass
class SynthCapture:
    images: list[np.ndarray]
    cameras: list[Camera]  # 真值相机
    K: np.ndarray
    shape: FootShape
    foot_to_paper: tuple[np.ndarray, np.ndarray]  # (R, t)
    gt_mesh: Mesh  # 纸面坐标系
    meta: dict = field(default_factory=dict)


def place_on_paper(yaw_deg: float, heel_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """脚坐标系 -> 纸面坐标系。脚轴大致沿纸的长边（纸面 +y）。"""
    a = np.radians(yaw_deg)
    base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 脚 x -> 纸 y
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return rz @ base, np.array([heel_xy[0], heel_xy[1], 0.0])


def orbit_cameras(
    K: np.ndarray,
    image_size: tuple[int, int],
    target: np.ndarray,
    n_views: int,
    rng: np.random.Generator,
    elevations_deg=DEFAULT_ELEVATIONS,
    distance_mm: float = 420.0,
) -> list[Camera]:
    """模拟手机绕脚若干圈（每圈一个仰角），带手持抖动。"""
    cams = []
    per_ring = [n_views // len(elevations_deg)] * len(elevations_deg)
    per_ring[-1] += n_views - sum(per_ring)
    for elev, n in zip(elevations_deg, per_ring):
        for i in range(n):
            az = np.radians(360.0 * i / n + rng.uniform(-3, 3))
            el = np.radians(elev + rng.uniform(-3, 3))
            dist = distance_mm + rng.uniform(-40, 40)
            c = target + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
            look = target + rng.uniform(-15, 15, 3) * np.array([1, 1, 0.5])
            cams.append(Camera.look_at(K, c, look, image_size, roll_deg=rng.uniform(-4, 4)))
    return cams


def _floor_texture(rng: np.random.Generator, size: int = 1600) -> np.ndarray:
    noise = rng.normal(0, 1, (size // 8, size // 8)).astype(np.float32)
    noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (size, size)).astype(np.float32), (0, 0), 1.5)
    gray = 68 + 12 * noise + 6 * fine
    tex = np.stack([gray * 0.95, gray, gray * 1.05], axis=2)
    return np.clip(tex, 0, 255).astype(np.uint8)


def render_view(
    cam: Camera,
    mesh: Mesh,
    floor_tex: np.ndarray | None,
    paper_mm=A4_MM,
    supersample: int = 2,
    color=SKIN_BGR,
    background=(70, 66, 62),
) -> np.ndarray:
    w, h = cam.image_size
    ss = supersample
    k = cam.K.copy()
    k[:2] *= ss
    k[:2, 2] += (ss - 1) / 2.0  # 像素中心对齐
    big = Camera(k, cam.R, cam.t, (w * ss, h * ss))
    img = np.empty((h * ss, w * ss, 3), np.uint8)
    img[:] = background

    # 地面：纹理通过平面单应性贴图，只画在地平线以下
    if floor_tex is not None:
        x0, y0 = -650.0, -650.0
        hom = k @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
        hom = hom @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
        floor = cv2.warpPerspective(floor_tex, hom, (w * ss, h * ss), flags=cv2.INTER_LINEAR)
        # 射线世界方向的 z 分量对像素坐标是线性的
        a, b, c = (cam.R.T @ np.linalg.inv(k))[2]
        below = (a * np.arange(w * ss)[None, :] + b * np.arange(h * ss)[:, None] + c) < -1e-3
        img[below] = floor[below]

    # 纸
    pw, ph = paper_mm
    corners = np.array([[0, 0, 0], [pw, 0, 0], [pw, ph, 0], [0, ph, 0]], float)
    uv, _ = big.project(corners)
    cv2.fillConvexPoly(img, np.round(uv * 8).astype(np.int32), PAPER_BGR.tolist(), cv2.LINE_8, 3)

    # 脚：背面剔除 + 画家算法 + 漫反射着色
    tri = mesh.triangles
    normals = mesh.face_normals()
    cent = tri.mean(axis=1)
    facing = np.einsum("ij,ij->i", normals, cent - cam.center) < 0
    tri, normals, cent = tri[facing], normals[facing], cent[facing]
    depth = big.to_camera(cent)[:, 2]
    order = np.argsort(-depth)
    uv, _ = big.project(tri.reshape(-1, 3))
    uv = np.round(uv.reshape(-1, 3, 2) * 8).astype(np.int32)
    light = np.array([0.35, -0.25, 1.0])
    light /= np.linalg.norm(light)
    shade = 0.38 + 0.62 * np.clip(normals @ light, 0, 1)
    cols = (np.asarray(color)[None] * shade[:, None]).clip(0, 255)
    for i in order:
        cv2.fillConvexPoly(img, uv[i], cols[i].tolist(), cv2.LINE_8, 3)

    if ss > 1:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def make_capture(
    seed: int = 0,
    n_views: int = 48,
    image_size: tuple[int, int] = (1440, 1080),
    focal_px: float = 1100.0,
    shape: FootShape | None = None,
    noise_sigma: float = 2.0,
    gt_spacing: float = 0.8,
    gt_z_max: float = 140.0,
    elevations_deg=DEFAULT_ELEVATIONS,
) -> SynthCapture:
    rng = np.random.default_rng(seed)
    shape = shape or (FootShape() if seed == 0 else FootShape.random(rng))
    K = intrinsics(focal_px, image_size)
    # 脚放在纸中间、整只脚都在纸上（A4 长 297 mm，脚长超过约 285 mm 需要换 A3）
    foot_len = 258.0 * shape.length_scale
    heel_y = (A4_MM[1] - foot_len) / 2 + rng.uniform(-1, 1) * max(0.0, (A4_MM[1] - foot_len) / 2 - 6)
    rot, trans = place_on_paper(rng.uniform(-5, 5), (105 + rng.uniform(-8, 8), heel_y))
    # 渲染用较粗的网格（含小腿），真值用细网格
    render_mesh = foot_mesh(shape, spacing=1.6).transformed(rot, trans)
    gt_mesh = foot_mesh(shape, spacing=gt_spacing, z_max=gt_z_max).transformed(rot, trans)
    target = np.array([A4_MM[0] / 2, A4_MM[1] / 2, 45.0])
    cams = orbit_cameras(K, image_size, target, n_views, rng, elevations_deg)
    tex = _floor_texture(rng)
    images = []
    for cam in cams:
        img = render_view(cam, render_mesh, tex).astype(np.float32)
        img += rng.normal(0, noise_sigma, img.shape)
        images.append(np.clip(img, 0, 255).astype(np.uint8))
    return SynthCapture(images, cams, K, shape, (rot, trans), gt_mesh, {"seed": seed})
