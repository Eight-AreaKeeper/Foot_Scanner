"""从脚的 3D 网格提取鞋楦相关测量值。

坐标约定：网格位于纸面坐标系（z = 0 为地面，单位 mm）。脚轴取低处（z < 25 mm）
顶点在地面上投影的主方向，脚跟一端由小腿所在位置判定。

各项定义（与 docs/tech-design.md 第 5 节对应，均为原型阶段的近似定义）：
- 脚长：沿脚轴，脚跟最后点到最前脚趾的距离。
- 跖围宽：脚长 60%–80% 区间内垂直于脚轴的最大宽度。
- 跖围：经过该区间内内侧、外侧最突出点（近似第一、第五跖趾关节）的竖直截面，
  截面凸包周长（相当于皮尺绕一圈）。
- 脚背高度：脚长 50% 处截面的最高点。
- 脚背围：脚长 50% 处、垂直于脚轴的竖直截面凸包周长。
- 脚跟宽：脚长 1/6 处、离地 30 mm 以下的截面宽度（避开踝骨）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .mesh import Mesh, hull_perimeter, section_points

MEASUREMENT_LABELS = {
    "length_mm": "脚长",
    "ball_width_mm": "跖围宽",
    "ball_girth_mm": "跖围",
    "instep_height_mm": "脚背高度",
    "instep_girth_mm": "脚背围",
    "heel_width_mm": "脚跟宽",
}


@dataclass
class FootFrame:
    heel: np.ndarray  # 脚跟最后点在地面上的位置 (x, y)
    axis: np.ndarray  # 脚轴单位向量（地面内，指向脚趾）
    side: np.ndarray  # 地面内垂直于脚轴的单位向量

    def to_local(self, pts: np.ndarray) -> np.ndarray:
        """返回 (u 沿脚轴, s 横向, z 高度)。"""
        rel = pts[:, :2] - self.heel
        return np.column_stack([rel @ self.axis, rel @ self.side, pts[:, 2]])

    def to_world(self, u: float, s: float, z: float = 0.0) -> np.ndarray:
        xy = self.heel + u * self.axis + s * self.side
        return np.array([xy[0], xy[1], z])


@dataclass
class FootMeasurements3D:
    length_mm: float
    ball_width_mm: float
    ball_girth_mm: float
    instep_height_mm: float
    instep_girth_mm: float
    heel_width_mm: float
    # 足弓高度需要看到脚底，visual hull 做不到；留给稠密重建 / LiDAR 版本
    arch_height_mm: float | None = None

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class MeasurementDetail:
    frame: FootFrame
    ball_section: np.ndarray  # (M, 2) 截面点（截面内坐标，z）
    instep_section: np.ndarray
    ball_points: tuple[np.ndarray, np.ndarray]  # 内外侧最突出点（世界坐标）


def foot_frame(mesh: Mesh) -> FootFrame:
    v = mesh.vertices
    low = v[v[:, 2] < 25.0, :2]
    c = low.mean(axis=0)
    _, _, vt = np.linalg.svd(low - c, full_matrices=False)
    axis = vt[0]
    high = v[v[:, 2] > 0.7 * v[:, 2].max(), :2]
    if len(high) and (high.mean(axis=0) - c) @ axis > 0:
        axis = -axis  # 小腿靠近脚跟
    body = v[v[:, 2] < 60.0, :2]
    heel_u = ((body - c) @ axis).min()
    side = np.array([-axis[1], axis[0]])
    return FootFrame(c + heel_u * axis, axis, side)


def _section_local(mesh: Mesh, frame: FootFrame, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    pts = section_points(mesh, point, normal)
    return frame.to_local(pts) if len(pts) else np.zeros((0, 3))


def measure_mesh(mesh: Mesh) -> tuple[FootMeasurements3D, MeasurementDetail]:
    frame = foot_frame(mesh)
    loc = frame.to_local(mesh.vertices)
    body = loc[loc[:, 2] < 60.0]
    length = float(body[:, 0].max())

    band = loc[(loc[:, 0] > 0.6 * length) & (loc[:, 0] < 0.8 * length)]
    i_min, i_max = np.argmin(band[:, 1]), np.argmax(band[:, 1])
    ball_width = float(band[i_max, 1] - band[i_min, 1])
    p1 = frame.to_world(band[i_min, 0], band[i_min, 1])
    p5 = frame.to_world(band[i_max, 0], band[i_max, 1])

    # 跖围：过 p1、p5 的竖直平面
    along = (p5 - p1)[:2] / np.linalg.norm((p5 - p1)[:2])
    normal = np.array([-along[1], along[0], 0.0])
    sec = section_points(mesh, p1, normal)
    t = (sec[:, :2] - p1[:2]) @ along
    keep = (t > -40) & (t < np.linalg.norm((p5 - p1)[:2]) + 40)
    ball_sec = np.column_stack([t[keep], sec[keep, 2]])
    ball_girth = hull_perimeter(ball_sec)

    # 脚背：脚长 50% 处横截面
    mid = frame.to_world(0.5 * length, 0.0)
    sec = _section_local(mesh, frame, mid, np.array([*frame.axis, 0.0]))
    instep_sec = sec[:, 1:3]
    instep_height = float(instep_sec[:, 1].max())
    instep_girth = hull_perimeter(instep_sec)

    # 脚跟宽
    heel_pt = frame.to_world(length / 6.0, 0.0)
    sec = _section_local(mesh, frame, heel_pt, np.array([*frame.axis, 0.0]))
    sec = sec[sec[:, 2] < 30.0]
    heel_width = float(sec[:, 1].max() - sec[:, 1].min())

    m = FootMeasurements3D(
        length_mm=length,
        ball_width_mm=ball_width,
        ball_girth_mm=ball_girth,
        instep_height_mm=instep_height,
        instep_girth_mm=instep_girth,
        heel_width_mm=heel_width,
    )
    return m, MeasurementDetail(frame, ball_sec, instep_sec, (p1, p5))
