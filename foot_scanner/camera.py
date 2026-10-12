"""从纸面单应性恢复相机位姿，用于视差校正。

脚的轮廓不在纸平面上（脚跟最后点离地约 2–3 cm，脚掌两侧约 1–2 cm），
从斜上方拍摄时，这些点投影到纸面会被"推远"，导致长宽偏大。知道相机在纸面
坐标系中的位置后，就能把轮廓点沿视线拉回到真实高度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# iPhone 主摄等效焦距约 26 mm；35 mm 画幅对角线 43.27 mm
DEFAULT_EQUIV_FOCAL_MM = 26.0
FULL_FRAME_DIAG_MM = 43.27


@dataclass
class CameraPose:
    focal_px: float
    focal_source: str  # "exif" | "estimated" | "default"
    center_mm: np.ndarray  # 相机光心在纸面坐标系中的位置 (x, y, z)，z 为离纸面高度

    @property
    def height_mm(self) -> float:
        return float(self.center_mm[2])

    @property
    def nadir_mm(self) -> np.ndarray:
        """相机正下方在纸面上的点。"""
        return self.center_mm[:2].copy()


def focal_from_equiv(equiv_mm: float, image_size: tuple[int, int]) -> float:
    w, h = image_size
    return equiv_mm / FULL_FRAME_DIAG_MM * float(np.hypot(w, h))


def estimate_focal(homography: np.ndarray, image_size: tuple[int, int]) -> float | None:
    """假设主点在图像中心、像素为正方形，由平面单应性的两个约束估计焦距。

    相机接近正俯视时两个约束都会退化，此时返回 None。
    """
    w, h = image_size
    t = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1]])
    hm = t @ homography
    h1, h2 = hm[:, 0], hm[:, 1]
    candidates = []
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        candidates.append((-(h1[0] * h2[0] + h1[1] * h2[1]) / den, abs(den)))
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        candidates.append((-(h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / den, abs(den)))
    long_side = max(w, h)
    good = [(f2, wgt) for f2, wgt in candidates if f2 > 0 and 0.4 * long_side < np.sqrt(f2) < 2.5 * long_side]
    if not good:
        return None
    # 取条件更好的那个约束（分母更大）
    f2 = max(good, key=lambda g: g[1])[0]
    return float(np.sqrt(f2))


def recover_pose(
    homography: np.ndarray,
    image_size: tuple[int, int],
    equiv_focal_mm: float | None = None,
) -> CameraPose:
    w, h = image_size
    if equiv_focal_mm:
        f, source = focal_from_equiv(equiv_focal_mm, image_size), "exif"
    else:
        est = estimate_focal(homography, image_size)
        if est is not None:
            f, source = est, "estimated"
        else:
            f, source = focal_from_equiv(DEFAULT_EQUIV_FOCAL_MM, image_size), "default"

    k = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    b = np.linalg.inv(k) @ homography
    lam = 2.0 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    if b[2, 2] < 0:  # 纸在相机前方：t_z 必须为正
        lam = -lam
    r1, r2, tvec = lam * b[:, 0], lam * b[:, 1], lam * b[:, 2]
    r3 = np.cross(r1, r2)
    r = np.column_stack([r1, r2, r3])
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    center = -r.T @ tvec
    if center[2] < 0:
        # 纸面坐标系 z 轴可能朝下，翻转使相机高度为正
        center[2] = -center[2]
    return CameraPose(focal_px=f, focal_source=source, center_mm=center)


def correct_parallax(points_mm: np.ndarray, heights_mm: np.ndarray, pose: CameraPose) -> np.ndarray:
    """把投影到纸面的轮廓点按其真实高度沿视线拉回。

    纸面上看到的点 p，若真实高度为 z，则真实水平位置为 N + (p - N)(1 - z / H)，
    其中 N 为相机正下方点，H 为相机高度。
    """
    n = pose.nadir_mm
    factor = 1.0 - np.asarray(heights_mm, dtype=np.float64) / pose.height_mm
    return n + (np.asarray(points_mm, dtype=np.float64) - n) * factor[:, None]
