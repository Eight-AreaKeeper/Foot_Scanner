import numpy as np
import pytest

from foot_scanner import synth
from foot_scanner.camera import recover_pose
from foot_scanner.measure import find_frame, measure
from foot_scanner.paper import PaperNotFound, detect_paper
from foot_scanner.pipeline import scan
from foot_scanner.segment import FootNotFound
from foot_scanner.sizing import recommend_sizes


def ground_truth(scene):
    return measure(scene.outline_mm, find_frame(scene.outline_mm))


def assert_close(result, gt, tol_mm):
    m = result.measurements
    assert m.length_mm == pytest.approx(gt.length_mm, abs=tol_mm)
    assert m.ball_width_mm == pytest.approx(gt.ball_width_mm, abs=tol_mm)
    assert m.heel_width_mm == pytest.approx(gt.heel_width_mm, abs=tol_mm)


@pytest.mark.parametrize(
    "camera, roll, foot_angle, foot",
    [
        ((105, 380, 600), 0, 0, {}),
        ((60, 420, 550), 12, 8, {"length": 245, "ball_width": 96, "heel_width": 62}),
        ((170, 330, 750), -20, -10, {"length": 285, "ball_width": 108, "heel_width": 70}),
        ((105, 200, 700), 90, 0, {"length": 230, "ball_width": 92, "heel_width": 60}),
    ],
)
def test_free_mode_matches_ground_truth(camera, roll, foot_angle, foot):
    scene = synth.render(camera_center_mm=camera, roll_deg=roll, foot_angle_deg=foot_angle, foot_kwargs=foot)
    result = scan(scene.image, equiv_focal_mm=scene.equiv_focal_mm)
    assert_close(result, ground_truth(scene), tol_mm=1.0)


def test_parallax_correction_matters():
    # 相机在脚正上方时，脚跟、脚尖都被往外推，误差叠加最明显
    scene = synth.render(camera_center_mm=(105, 150, 500))
    gt = ground_truth(scene)
    raw = scan(scene.image, equiv_focal_mm=26, parallax=False).measurements
    corrected = scan(scene.image, equiv_focal_mm=26).measurements
    assert raw.length_mm - gt.length_mm > 5  # 不校正会偏大
    assert abs(corrected.length_mm - gt.length_mm) < abs(raw.length_mm - gt.length_mm)


def test_focal_estimated_without_exif():
    scene = synth.render(camera_center_mm=(60, 450, 500), roll_deg=10)
    result = scan(scene.image)  # 不给焦距
    assert result.pose.focal_source == "estimated"
    assert_close(result, ground_truth(scene), tol_mm=1.5)


def test_pose_recovery():
    scene = synth.render(camera_center_mm=(80, 400, 650), roll_deg=5)
    h, w = scene.image.shape[:2]
    paper = detect_paper(scene.image)
    pose = recover_pose(paper.homography, (w, h), scene.equiv_focal_mm)
    # 纸的四个角对称，朝向可能差 180°，比较相机到纸中心的距离和高度
    center = np.array([105.0, 148.5])
    assert pose.height_mm == pytest.approx(650, abs=5)
    assert np.linalg.norm(pose.nadir_mm - center) == pytest.approx(np.linalg.norm([80 - 105, 400 - 148.5]), abs=5)


def test_wall_mode():
    scene = synth.render(mode="wall", camera_center_mm=(105, 330, 600), target_mm=(105, 140, 0))
    result = scan(scene.image, mode="wall", equiv_focal_mm=26)
    # 靠墙模式下脚跟就在纸边上，真实脚长 = 轮廓最远点到纸边
    true_len = scene.outline_mm[:, 1].max()
    assert result.measurements.length_mm == pytest.approx(true_len, abs=1.0)


def test_no_paper_raises():
    img = np.full((1200, 900, 3), 70, np.uint8)
    with pytest.raises(PaperNotFound):
        scan(img)


def test_empty_paper_raises():
    scene = synth.render(foot_kwargs={"length": 1, "ball_width": 1, "heel_width": 1})
    with pytest.raises(FootNotFound):
        scan(scene.image)


def test_sizes():
    s = recommend_sizes(258.0)
    assert s["cm"] == 26.0
    assert s["cn"] == 260
    assert s["us_men"] == 8.0
    assert s["eu"] == 41.5
    assert recommend_sizes(261.0)["cm"] == 26.5  # 向上取半码
