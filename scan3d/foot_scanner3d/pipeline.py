"""完整流程：多帧照片 -> 位姿 -> 雕刻重建 -> 测量。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .carve import View, carve, outside_fraction, robust_hull_points
from .measure import FootMeasurements3D, MeasurementDetail, measure_mesh
from .mesh import Mesh
from .paper import A4_MM, PaperFit, PaperNotFound, fit_paper, resolve_symmetry
from .segment import segment_foot, signed_distance


@dataclass
class ScanResult:
    mesh: Mesh
    measurements: FootMeasurements3D
    detail: MeasurementDetail
    cameras: list[Camera | None]
    fits: list[PaperFit | None]
    masks: list[np.ndarray]
    rejected: dict[int, str] = field(default_factory=dict)

    @property
    def n_used(self) -> int:
        return sum(c is not None for c in self.cameras)


def reconstruct(
    images: list[np.ndarray],
    K: np.ndarray,
    paper_mm=A4_MM,
    voxel_mm: float = 1.0,
    max_height_mm: float = 140.0,
    min_views: int = 12,
) -> ScanResult:
    """images 需按拍摄顺序排列（用于消除 A4 的 180° 对称歧义）。"""
    masks, fits, rejected = [], [], {}
    for i, img in enumerate(images):
        mask = segment_foot(img)
        masks.append(mask)
        try:
            fits.append(fit_paper(img, mask, K, paper_mm))
        except PaperNotFound as e:
            fits.append(None)
            rejected[i] = str(e)
    cams = resolve_symmetry(fits)
    sdfs = [signed_distance(m) if f is not None else None for m, f in zip(masks, fits)]
    pw, ph = paper_mm
    lo = np.array([-30.0, -30.0, 0.0])
    hi = np.array([pw + 30.0, ph + 30.0, max_height_mm])
    cams = _consistency_check(cams, fits, sdfs, lo, hi, rejected)

    views = [View(c, sdf) for c, sdf in zip(cams, sdfs) if c is not None]
    if len(views) < min_views:
        raise RuntimeError(f"可用帧只有 {len(views)} 张（至少 {min_views}）：{rejected}")
    mesh = carve(views, lo, hi, voxel=voxel_mm)
    meas, detail = measure_mesh(mesh)
    return ScanResult(mesh, meas, detail, cams, fits, masks, rejected)


def _consistency_check(cams, fits, sdfs, lo, hi, rejected, max_outside: float = 0.02, rounds: int = 2):
    """多帧一致性检查（相当于 App 端的"这几张要重拍"）。

    先用稳健 hull（容忍少数坏帧）得到脚的大致形状，再逐帧检查：两个 180° 对称解里
    选和 hull 吻合的那个；都对不上的帧（纸边拟合错、分割失败）丢弃。
    """
    cams = list(cams)
    for _ in range(rounds):
        active = [View(c, s) for c, s in zip(cams, sdfs) if c is not None]
        if len(active) < 4:
            break
        hull = robust_hull_points(active, lo, hi, outliers=max(1, round(0.12 * len(active))))
        changed = False
        for i, fit in enumerate(fits):
            if fit is None:
                continue
            score, cam = min(
                ((outside_fraction(View(c, sdfs[i]), hull), c) for _, c in fit.candidates), key=lambda sc: sc[0]
            )
            new = cam if score <= max_outside else None
            if new is not cams[i]:
                changed = True
            cams[i] = new
            if new is None:
                rejected[i] = "与其他帧不一致（%.0f%% 的 hull 落在轮廓外）" % (100 * score)
            else:
                rejected.pop(i, None)
        if not changed:
            break
    return cams


def pose_errors(est: list[Camera | None], gt: list[Camera]) -> dict:
    """相机中心误差（mm）和朝向误差（度）。估计结果可能和真值差一个 180° 旋转。"""
    pairs = [(e, g) for e, g in zip(est, gt) if e is not None]
    if not pairs:
        return {}
    pw, ph = A4_MM
    flip = np.diag([-1.0, -1.0, 1.0])
    shift = np.array([pw, ph, 0.0])

    def errs(flipped: bool):
        pos, rot = [], []
        for e, g in pairs:
            c = e.center
            r = e.R
            if flipped:
                c = flip @ c + shift
                r = r @ flip
            pos.append(np.linalg.norm(c - g.center))
            cosang = (np.trace(r @ g.R.T) - 1) / 2
            rot.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        return np.array(pos), np.array(rot)

    pos, rot = min((errs(False), errs(True)), key=lambda x: x[0].mean())
    return {
        "center_mean_mm": float(pos.mean()),
        "center_max_mm": float(pos.max()),
        "rot_mean_deg": float(rot.mean()),
        "rot_max_deg": float(rot.max()),
    }
