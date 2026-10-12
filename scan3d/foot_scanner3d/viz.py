"""调试图：输入帧检测结果、重建网格、截面对比、测量表。"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera, intrinsics
from .measure import FootMeasurements3D, MeasurementDetail
from .mesh import Mesh
from .paper import PaperFit
from .synth import render_view

PANEL = (720, 540)
_LABELS_EN = {
    "length_mm": "Length",
    "ball_width_mm": "Ball width",
    "ball_girth_mm": "Ball girth",
    "instep_height_mm": "Instep height",
    "instep_girth_mm": "Instep girth",
    "heel_width_mm": "Heel width",
}


def _fit_panel(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, PANEL, interpolation=cv2.INTER_AREA)


def draw_detection(image: np.ndarray, mask: np.ndarray, fit: PaperFit | None) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 120, 0), 2)
    if fit is not None:
        for x, y in fit.edge_points[::3]:
            cv2.circle(out, (int(x), int(y)), 1, (0, 200, 0), -1)
        q = fit.corners_px
        for i in range(4):
            a, b = q[i], q[(i + 1) % 4]
            cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(out, tuple(map(int, a)), 7, (0, 0, 255), 2)
    return _fit_panel(out)


def draw_mesh(mesh: Mesh, image_size=(1440, 1080)) -> np.ndarray:
    c = mesh.vertices.mean(axis=0)
    c[2] = 40.0
    cam = Camera.look_at(intrinsics(1100.0, image_size), c + np.array([-260.0, -230.0, 260.0]), c, image_size)
    img = render_view(cam, mesh, None, color=np.array([200.0, 200.0, 200.0]), background=(40, 40, 40))
    return _fit_panel(img)


def draw_sections(rec: MeasurementDetail, gt: MeasurementDetail | None) -> np.ndarray:
    img = np.full((PANEL[1], PANEL[0], 3), 255, np.uint8)
    scale = 2.2  # px / mm
    step = 10 * scale
    for x in np.arange(0, PANEL[0], step):
        cv2.line(img, (int(x), 100), (int(x), 400), (235, 235, 235), 1)
    for y in np.arange(400, 99, -step):
        cv2.line(img, (0, int(y)), (PANEL[0], int(y)), (235, 235, 235), 1)

    def plot(sec2d, origin, color):
        if len(sec2d) == 0:
            return
        c = np.array([np.median(sec2d[:, 0]), 0.0])
        pts = (sec2d - c) * np.array([scale, -scale]) + origin
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 1, color, -1)

    for title, attr, ox in (("Ball section", "ball_section", 180), ("Instep section", "instep_section", 540)):
        origin = np.array([ox, 400.0])
        cv2.line(img, (ox - 160, 400), (ox + 160, 400), (180, 180, 180), 1)
        if gt is not None:
            plot(getattr(gt, attr), origin, (0, 160, 0))
        plot(getattr(rec, attr), origin, (0, 0, 220))
        cv2.putText(img, title, (ox - 70, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "red: reconstruction", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)
    if gt is not None:
        cv2.putText(img, "green: ground truth", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 160, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "grid: 10 mm", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def draw_table(rec: FootMeasurements3D, gt: FootMeasurements3D | None, extra: list[str]) -> np.ndarray:
    img = np.full((PANEL[1], PANEL[0], 3), 255, np.uint8)
    put = lambda s, y, x=30, col=(0, 0, 0): cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)
    put("Measurement", 50)
    put("Recon", 50, 300)
    if gt is not None:
        put("Truth", 50, 430)
        put("Error", 50, 560)
    r, g = rec.as_dict(), gt.as_dict() if gt is not None else None
    y = 90
    for k, label in _LABELS_EN.items():
        put(label, y)
        put(f"{r[k]:.1f}", y, 300)
        if g is not None:
            err = r[k] - g[k]
            put(f"{g[k]:.1f}", y, 430)
            put(f"{err:+.1f}", y, 560, (0, 140, 0) if abs(err) <= 2 else (0, 0, 220))
        y += 40
    put("Arch height: n/a (needs dense recon / LiDAR)", y + 10, col=(120, 120, 120))
    y += 60
    for line in extra:
        put(line, y, col=(80, 80, 80))
        y += 32
    return img


def debug_montage(
    image: np.ndarray,
    mask: np.ndarray,
    fit: PaperFit | None,
    mesh: Mesh,
    rec: tuple[FootMeasurements3D, MeasurementDetail],
    gt: tuple[FootMeasurements3D, MeasurementDetail] | None,
    extra: list[str],
) -> np.ndarray:
    a = draw_detection(image, mask, fit)
    b = draw_mesh(mesh)
    c = draw_sections(rec[1], gt[1] if gt else None)
    d = draw_table(rec[0], gt[0] if gt else None, extra)
    return np.vstack([np.hstack([a, b]), np.hstack([c, d])])
