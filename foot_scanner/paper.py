"""参照纸（A4 等）检测与透视校正。

思路：纸是画面中最大的一块亮、低饱和区域。脚踩在纸中间不会改变这块区域的
凸包，所以只要四个角露在画面里，就能从凸包拟合出四边形并求单应性矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAPER_SIZES_MM = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
    "a3": (297.0, 420.0),
}


class PaperNotFound(RuntimeError):
    pass


@dataclass
class Paper:
    corners_px: np.ndarray  # (4, 2) 图像坐标，顺序：左上、右上、右下、左下（纸面坐标系下）
    size_mm: tuple[float, float]  # (宽, 高)，高为长边
    homography: np.ndarray  # 纸面毫米坐标 -> 图像像素

    def image_to_paper(self, pts_px: np.ndarray) -> np.ndarray:
        """图像像素坐标 -> 纸面毫米坐标（只对纸平面上的点严格成立）。"""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)
        inv = np.linalg.inv(self.homography)
        return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)

    def warp(self, image: np.ndarray, px_per_mm: float) -> np.ndarray:
        """把纸面区域校正成俯视图，每毫米 px_per_mm 个像素。"""
        w_mm, h_mm = self.size_mm
        scale = np.diag([1.0 / px_per_mm, 1.0 / px_per_mm, 1.0])
        m = self.homography @ scale  # 俯视图像素 -> 原图像素
        size = (int(round(w_mm * px_per_mm)), int(round(h_mm * px_per_mm)))
        return cv2.warpPerspective(
            image, m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """把四个角排成首尾相接的顺序，且第一条边是短边（纸宽方向）。"""
    quad = quad.reshape(4, 2).astype(np.float64)
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]  # 图像坐标下顺时针
    edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
    if edges[0] + edges[2] > edges[1] + edges[3]:
        quad = np.roll(quad, -1, axis=0)
    # 让第一个点是图像中较靠上的那个短边端点，结果更稳定、便于调试
    if quad[0, 1] + quad[1, 1] > quad[2, 1] + quad[3, 1]:
        quad = np.roll(quad, 2, axis=0)
    return quad


def _hull_to_quad(hull: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.08, 40):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        if len(approx) < 4:
            break
    return None


def _refine_with_lines(hull: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """用轮廓上靠近每条边的点做直线拟合，再求交点，得到亚像素级角点。"""
    pts = hull.reshape(-1, 2).astype(np.float64)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        length = np.linalg.norm(ab)
        if length < 1:
            return quad
        d = ab / length
        n = np.array([-d[1], d[0]])
        rel = pts - a
        t = rel @ d
        dist = np.abs(rel @ n)
        sel = (t > 0.1 * length) & (t < 0.9 * length) & (dist < max(3.0, 0.01 * length))
        if sel.sum() < 2:
            lines.append((a, d))
            continue
        seg = pts[sel]
        vx, vy, x0, y0 = cv2.fitLine(seg.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], np.float64), np.array([vx, vy], np.float64)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        a = np.array([d1, -d2]).T
        if abs(np.linalg.det(a)) < 1e-9:
            out.append(quad[i])
            continue
        s, _ = np.linalg.solve(a, p2 - p1)
        out.append(p1 + s * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - quad, axis=1)) > 0.05 * cv2.arcLength(quad.astype(np.float32), True):
        return quad
    return out


def detect_paper(image: np.ndarray, paper: str = "a4") -> Paper:
    if paper not in PAPER_SIZES_MM:
        raise ValueError(f"未知纸张尺寸: {paper}")
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 0)
    s = hsv[:, :, 1]
    _, bright = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((bright > 0) & (s < 80)).astype(np.uint8) * 255
    k = max(3, int(min(h, w) * 0.004) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < 0.03 * h * w:
            break
        hull = cv2.convexHull(c)
        quad = _hull_to_quad(hull)
        if quad is None:
            continue
        quad_area = cv2.contourArea(quad.astype(np.float32))
        fill = cv2.contourArea(c) / max(quad_area, 1)
        if fill < 0.5 or quad_area > 0.9 * h * w:
            continue
        # 纸必须明显比周围亮，否则 Otsu 只是把均匀画面一分为二
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [quad.astype(np.int32)], 255)
        ring = cv2.dilate(inside, np.ones((k * 9, k * 9), np.uint8)) & ~inside
        if ring.any() and np.median(v[inside > 0]) - np.median(v[ring > 0]) < 40:
            continue
        best = (c, quad)
        break
    if best is None:
        raise PaperNotFound("没有找到参照纸：请确认整张纸和四个角都在画面里，且纸下是深色或有纹理的地面")

    contour, quad = best
    quad = _order_corners(quad)
    # 用原始轮廓（而不是稀疏的凸包顶点）做边线拟合
    quad = _order_corners(_refine_with_lines(contour, quad))

    w_mm, h_mm = PAPER_SIZES_MM[paper]
    dst = quad.astype(np.float64)
    src = np.array([[0, 0], [w_mm, 0], [w_mm, h_mm], [0, h_mm]], dtype=np.float64)
    homography, _ = cv2.findHomography(src, dst)
    return Paper(corners_px=quad, size_mm=(w_mm, h_mm), homography=homography)
