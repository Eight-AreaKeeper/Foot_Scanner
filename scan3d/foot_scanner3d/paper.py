"""从单帧中定位 A4 纸并求相机位姿。

绕脚拍摄时，脚和小腿经常挡住纸的一两个角，所以不依赖四个角都可见：
只取纸和地面之间的边界（亚像素等值线），用 RANSAC 拟合四条直线，
再求交点得到四个角（被挡住的角也能推算出来），最后用已知内参做 PnP。

A4 纸有 180° 对称，单帧无法区分两种朝向；在连续帧之间选让相机移动最小的那一个。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import find_contours

from .camera import Camera

A4_MM = (210.0, 297.0)


class PaperNotFound(RuntimeError):
    pass


@dataclass
class PaperFit:
    corners_px: np.ndarray  # (4, 2) 按图像中的环绕顺序
    lines: list[tuple[np.ndarray, np.ndarray]]  # (点, 方向)
    edge_points: np.ndarray  # 参与拟合的边界点
    candidates: list[tuple[float, Camera]]  # (重投影误差, 相机)，两个 180° 对称解


def _paper_mask(image: np.ndarray, foot_mask: np.ndarray) -> tuple[np.ndarray, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((v > thr) & (hsv[:, :, 1] < 50) & (foot_mask == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise PaperNotFound("画面中没有明显的白纸")
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[keep, cv2.CC_STAT_AREA] < 0.01 * mask.size:
        raise PaperNotFound("白纸面积太小")
    return (labels == keep).astype(np.uint8), float(thr)


def _edge_points(image: np.ndarray, foot_mask: np.ndarray) -> np.ndarray:
    paper, _ = _paper_mask(image, foot_mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
    ring = cv2.dilate(paper, np.ones((25, 25), np.uint8)) & (1 - paper) & (foot_mask == 0)
    level = 0.5 * (np.median(gray[paper > 0]) + np.median(gray[ring > 0]))

    # 只在纸边附近找等值线，且远离脚和画面边缘
    near_edge = cv2.dilate(paper, np.ones((7, 7), np.uint8)) & (1 - cv2.erode(paper, np.ones((7, 7), np.uint8)))
    bad = cv2.dilate((foot_mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
    h, w = gray.shape
    border = np.zeros_like(bad)
    border[:6, :] = border[-6:, :] = border[:, :6] = border[:, -6:] = 1
    ok = near_edge & (1 - bad) & (1 - border)

    pts = []
    for c in find_contours(gray, level):
        xy = c[:, ::-1]
        ix = np.clip(np.round(xy).astype(int), 0, [w - 1, h - 1])
        pts.append(xy[ok[ix[:, 1], ix[:, 0]] > 0])
    pts = np.concatenate(pts) if pts else np.zeros((0, 2))
    if len(pts) < 100:
        raise PaperNotFound("纸的边缘可见部分太少")
    return pts


def _fit_lines(pts: np.ndarray, n_lines: int = 4, rng=None) -> list[tuple[np.ndarray, np.ndarray, int]]:
    rng = rng or np.random.default_rng(0)
    lines = []
    remaining = pts
    for _ in range(n_lines):
        if len(remaining) < 40:
            break
        best = None
        for _ in range(400):
            i, j = rng.choice(len(remaining), 2, replace=False)
            d = remaining[j] - remaining[i]
            length = np.linalg.norm(d)
            if length < 30:
                continue
            n = np.array([-d[1], d[0]]) / length
            inl = np.abs((remaining - remaining[i]) @ n) < 1.5
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 40:
            break
        sel = remaining[best]
        for tol in (1.0, 0.7):
            c = sel.mean(axis=0)
            _, _, vt = np.linalg.svd(sel - c)
            dist = np.abs((remaining - c) @ vt[1])
            sel = remaining[dist < tol]
        lines.append((c, vt[0], int(len(sel))))
        remaining = remaining[dist > 4.0]
    return lines


def _intersect(l1, l2) -> np.ndarray:
    (p1, d1), (p2, d2) = l1[:2], l2[:2]
    a = np.array([d1, -d2]).T
    s, _ = np.linalg.solve(a, p2 - p1)
    return p1 + s * d1


def _quad_from_lines(lines) -> np.ndarray:
    ang = [np.arctan2(d[1], d[0]) % np.pi for _, d, _ in lines]

    def diff(a, b):
        x = abs(a - b) % np.pi
        return min(x, np.pi - x)

    pairings = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    (a0, a1), (b0, b1) = min(pairings, key=lambda p: diff(ang[p[0][0]], ang[p[0][1]]) + diff(ang[p[1][0]], ang[p[1][1]]))
    L = lines
    return np.array([_intersect(L[a0], L[b0]), _intersect(L[b0], L[a1]), _intersect(L[a1], L[b1]), _intersect(L[b1], L[a0])])


def fit_paper(image: np.ndarray, foot_mask: np.ndarray, K: np.ndarray, paper_mm=A4_MM, max_reproj_px: float = 4.0) -> PaperFit:
    pts = _edge_points(image, foot_mask)
    lines = _fit_lines(pts)
    if len(lines) < 4:
        raise PaperNotFound("只找到 %d 条纸边" % len(lines))
    quad = _quad_from_lines(lines)
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        raise PaperNotFound("纸边拟合结果不是凸四边形")

    pw, ph = paper_mm
    obj = np.array([[0, 0, 0], [pw, 0, 0], [pw, ph, 0], [0, ph, 0]], np.float64)
    h, w = image.shape[:2]
    sols = []
    for oi, order in enumerate((quad, quad[::-1])):
        for shift in range(4):
            img_pts = np.roll(order, -shift, axis=0)
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
            if not ok:
                continue
            rmat = cv2.Rodrigues(rvec)[0]
            cam = Camera(K, rmat, tvec.ravel(), (w, h))
            if cam.center[2] <= 0:
                continue
            err = float(np.sqrt(np.mean(np.sum((cam.project(obj)[0] - img_pts) ** 2, axis=1))))
            sols.append((err, cam, oi, shift))
    if not sols:
        raise PaperNotFound("位姿求解失败")
    sols.sort(key=lambda s: s[0])
    if sols[0][0] > max_reproj_px:
        raise PaperNotFound("四边形与 A4 比例不符（重投影误差 %.1f px）" % sols[0][0])
    best = sols[0]
    # 180° 对称解：同一环绕方向、起点错开两个角
    twin = next((s for s in sols if s[2] == best[2] and s[3] == (best[3] + 2) % 4), best)
    return PaperFit(quad, [(c, d) for c, d, _ in lines], pts, [best[:2], twin[:2]])


def resolve_symmetry(fits: list[PaperFit | None]) -> list[Camera | None]:
    """按拍摄顺序选择 180° 对称解，使相邻帧相机位置连续。"""
    out: list[Camera | None] = []
    prev = None
    for fit in fits:
        if fit is None:
            out.append(None)
            continue
        if prev is None:
            cam = fit.candidates[0][1]
        else:
            cam = min((c for _, c in fit.candidates), key=lambda c: np.linalg.norm(c.center - prev))
        out.append(cam)
        prev = cam.center
    return out
