"""完整流程：照片 -> 参照纸 -> 脚轮廓 -> 视差校正 -> 测量 -> 鞋码。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraPose, correct_parallax, recover_pose
from .measure import DEFAULT_EDGE_HEIGHT_PROFILE, Measurements, edge_heights, find_frame, measure
from .paper import Paper, detect_paper
from .segment import FootMask, segment_foot
from .sizing import recommend_sizes

WARP_PX_PER_MM = 4.0
MIN_CAMERA_HEIGHT_MM = 350.0
MAX_NADIR_AHEAD = 0.6  # 相机正下方点超过脚长 60% 处（靠近脚尖）就提示


@dataclass
class ScanResult:
    measurements: Measurements
    sizes: dict
    paper: Paper
    pose: CameraPose
    foot: FootMask
    contour_raw_mm: np.ndarray
    contour_mm: np.ndarray
    warped: np.ndarray
    mode: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        m = self.measurements
        return {
            "mode": self.mode,
            "length_mm": round(m.length_mm, 1),
            "ball_width_mm": round(m.ball_width_mm, 1),
            "ball_position_pct": round(m.ball_position * 100, 1),
            "heel_width_mm": round(m.heel_width_mm, 1),
            "width_to_length": round(m.ball_width_mm / m.length_mm, 3),
            "sizes": self.sizes,
            "camera": {
                "height_mm": round(self.pose.height_mm, 0),
                "focal_px": round(self.pose.focal_px, 1),
                "focal_source": self.pose.focal_source,
            },
            "warnings": self.warnings,
        }


def read_equiv_focal(path: str) -> float | None:
    """从 EXIF 读 35 mm 等效焦距；没有 Pillow 或没有该字段时返回 None。"""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif().get_ifd(0x8769)
    except Exception:
        return None
    value = exif.get(41989)  # FocalLengthIn35mmFilm
    return float(value) if value else None


def scan(
    image: np.ndarray,
    paper_size: str = "a4",
    mode: str = "free",
    equiv_focal_mm: float | None = None,
    parallax: bool = True,
    edge_height_profile=DEFAULT_EDGE_HEIGHT_PROFILE,
) -> ScanResult:
    """mode="free"：脚放在纸中间；mode="wall"：纸的短边贴墙，脚跟贴墙。"""
    if mode not in ("free", "wall"):
        raise ValueError(f"未知模式: {mode}")
    warnings: list[str] = []
    h, w = image.shape[:2]
    paper = detect_paper(image, paper_size)
    pose = recover_pose(paper.homography, (w, h), equiv_focal_mm)

    warped = paper.warp(image, WARP_PX_PER_MM)
    foot = segment_foot(warped, WARP_PX_PER_MM)
    contour_raw = (foot.contour_px + 0.5) / WARP_PX_PER_MM  # 俯视图像素中心 -> 纸面毫米
    pw, ph = paper.size_mm

    axis = heel_x = None
    if mode == "wall":
        if foot.touches_edge["top"] == foot.touches_edge["bottom"]:
            raise ValueError("靠墙模式下脚跟应贴住纸的一条短边，而脚尖不能超出另一条短边")
        if foot.touches_edge["top"]:
            axis, heel_x = np.array([0.0, 1.0]), 0.0
        else:
            axis, heel_x = np.array([0.0, -1.0]), -ph
        if foot.touches_edge["left"] or foot.touches_edge["right"]:
            warnings.append("脚碰到了纸的长边，宽度可能被截断")
    else:
        touched = [k for k, v in foot.touches_edge.items() if v]
        if touched:
            warnings.append(f"脚碰到了纸边（{', '.join(touched)}），超出纸面的部分量不到，结果可能偏小")

    frame = find_frame(contour_raw, axis=axis, heel_x=heel_x)
    contour = contour_raw
    if parallax:
        # 迭代两次：先用未校正的长度估计各点高度，校正后再更新
        for _ in range(2):
            length = frame.to_local(contour)[:, 0].max()
            z = edge_heights(contour_raw, frame, length, edge_height_profile)
            contour = correct_parallax(contour_raw, z, pose)
            frame = find_frame(contour, axis=axis, heel_x=heel_x)

    if mode == "wall":
        # 贴墙的那一段轮廓被纸边截断，脚跟位置就是纸边，不参与视差校正
        local = frame.to_local(contour)
        raw_local = frame.to_local(contour_raw)
        near_wall = raw_local[:, 0] < 1.0
        local[near_wall, 0] = raw_local[near_wall, 0]
        contour = frame.to_paper(local)

    m = measure(contour, frame)

    if pose.height_mm < MIN_CAMERA_HEIGHT_MM:
        warnings.append(f"相机离地只有约 {pose.height_mm:.0f} mm，透视误差较大，建议举高到 50 cm 以上")
    if mode == "free":
        # 小腿的投影总是背离相机正下方点；相机越靠脚尖，小腿越会被"甩"到脚跟后面
        nadir_x = frame.to_local(pose.nadir_mm[None])[0, 0]
        if nadir_x > MAX_NADIR_AHEAD * m.length_mm:
            warnings.append("相机在脚尖一侧，小腿可能在画面里盖住脚跟后方，脚长可能偏大；请在脚踝正上方拍")
    if pose.focal_source == "default":
        warnings.append("无法确定相机焦距，视差校正使用了默认值")
    if not (0.33 <= m.ball_width_mm / m.length_mm <= 0.45):
        warnings.append("宽长比超出常见范围，请检查分割结果（阴影、裤腿、袜子颜色）")

    return ScanResult(
        measurements=m,
        sizes=recommend_sizes(m.length_mm),
        paper=paper,
        pose=pose,
        foot=foot,
        contour_raw_mm=contour_raw,
        contour_mm=contour,
        warped=warped,
        mode=mode,
        warnings=warnings,
    )


def render_debug(result: ScanResult) -> np.ndarray:
    """在纸面俯视图上画出轮廓、脚轴和测量线。"""
    s = WARP_PX_PER_MM
    img = result.warped.copy()

    def px(pts):
        return np.round(np.asarray(pts) * s).astype(np.int32)

    overlay = img.copy()
    overlay[result.foot.mask > 0] = (0.5 * overlay[result.foot.mask > 0] + [0, 0, 127]).astype(np.uint8)
    img = overlay
    cv2.polylines(img, [px(result.contour_raw_mm)], True, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [px(result.contour_mm)], True, (0, 200, 0), 2, cv2.LINE_AA)

    m = result.measurements
    f = m.frame
    heel, toe = f.to_paper(np.array([[0.0, 0.0], [m.length_mm, 0.0]]))
    cv2.line(img, tuple(px(heel)), tuple(px(toe)), (255, 0, 0), 2, cv2.LINE_AA)

    local = f.to_local(result.contour_mm)

    def width_line(t, color):
        x = t * m.length_mm
        band = np.abs(local[:, 0] - x) < 1.5
        if not band.any():
            return
        y0, y1 = local[band, 1].min(), local[band, 1].max()
        a, b = f.to_paper(np.array([[x, y0], [x, y1]]))
        cv2.line(img, tuple(px(a)), tuple(px(b)), color, 2, cv2.LINE_AA)

    width_line(m.ball_position, (0, 140, 255))
    width_line(0.16, (255, 0, 255))

    lines = [
        f"L {m.length_mm:.1f} mm",
        f"Ball W {m.ball_width_mm:.1f} mm",
        f"Heel W {m.heel_width_mm:.1f} mm",
        f"Cam H {result.pose.height_mm:.0f} mm",
    ]
    for i, text in enumerate(lines):
        y = 30 + 28 * i
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return img
