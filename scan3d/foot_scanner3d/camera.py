"""针孔相机：世界坐标（纸面坐标系，毫米，Z 轴朝上）-> 图像像素。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# iPhone 主摄等效焦距约 26 mm；35 mm 画幅对角线 43.27 mm
DEFAULT_EQUIV_FOCAL_MM = 26.0
FULL_FRAME_DIAG_MM = 43.27


def intrinsics(focal_px: float, image_size: tuple[int, int]) -> np.ndarray:
    w, h = image_size
    return np.array([[focal_px, 0, w / 2.0], [0, focal_px, h / 2.0], [0, 0, 1.0]])


def focal_from_equiv(equiv_mm: float, image_size: tuple[int, int]) -> float:
    w, h = image_size
    return equiv_mm / FULL_FRAME_DIAG_MM * float(np.hypot(w, h))


@dataclass
class Camera:
    K: np.ndarray  # 3x3 内参
    R: np.ndarray  # 3x3 世界 -> 相机
    t: np.ndarray  # (3,)
    image_size: tuple[int, int]  # (宽, 高)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def to_camera(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.R.T + self.t

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (像素坐标 (N,2), 深度 (N,))。"""
        pc = self.to_camera(pts)
        z = pc[:, 2]
        uv = pc[:, :2] / z[:, None] * np.array([self.K[0, 0], self.K[1, 1]]) + self.K[:2, 2]
        return uv, z

    @staticmethod
    def look_at(K, center, target, image_size, roll_deg: float = 0.0) -> "Camera":
        center = np.asarray(center, float)
        fwd = np.asarray(target, float) - center
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        r = np.stack([right, down, fwd])
        if roll_deg:
            a = np.radians(roll_deg)
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
            r = rz @ r
        return Camera(np.asarray(K, float), r, -r @ center, image_size)
