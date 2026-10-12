import numpy as np
import pytest

from foot_scanner3d.measure import measure_mesh
from foot_scanner3d.paper import fit_paper
from foot_scanner3d.pipeline import pose_errors, reconstruct
from foot_scanner3d.segment import segment_foot
from foot_scanner3d.synth import make_capture


@pytest.fixture(scope="module")
def capture():
    return make_capture(seed=3, n_views=32, gt_spacing=1.0)


@pytest.fixture(scope="module")
def result(capture):
    return reconstruct(capture.images, capture.K, voxel_mm=1.5)


def test_paper_pose_from_partially_occluded_paper(capture):
    # 第一帧来自贴地一圈：纸的远边被脚和小腿挡住一大段
    img = capture.images[0]
    fit = fit_paper(img, segment_foot(img), capture.K)
    errs = [pose_errors([c], [capture.cameras[0]])["center_mean_mm"] for _, c in fit.candidates]
    assert min(errs) < 2.0


def test_all_views_posed_consistently(capture, result):
    assert result.n_used >= len(capture.images) - 2
    pe = pose_errors(result.cameras, capture.cameras)
    assert pe["center_mean_mm"] < 2.0
    assert pe["rot_max_deg"] < 1.0


def test_measurements_match_ground_truth(capture, result):
    gt = measure_mesh(capture.gt_mesh)[0].as_dict()
    rec = result.measurements.as_dict()
    tol = {
        "length_mm": 1.5,
        "ball_width_mm": 1.5,
        "heel_width_mm": 1.5,
        "instep_height_mm": 2.0,
        "ball_girth_mm": 3.0,
        "instep_girth_mm": 3.0,
    }
    for key, t in tol.items():
        assert abs(rec[key] - gt[key]) <= t, (key, rec[key], gt[key])
    assert rec["arch_height_mm"] is None


def test_ground_truth_is_a_plausible_adult_foot(capture):
    gt = measure_mesh(capture.gt_mesh)[0]
    assert 220 < gt.length_mm < 300
    assert 85 < gt.ball_width_mm < 120
    assert 200 < gt.ball_girth_mm < 290


def test_inconsistent_view_is_rejected(capture):
    # 模拟一帧分割失败：脚的前半截被涂成地面颜色，这帧会把真实的脚雕掉
    images = [im.copy() for im in capture.images]
    bad = 5
    mask = segment_foot(images[bad])
    ys, xs = np.nonzero(mask)
    x_mid = int(np.median(xs))
    region = mask[:, x_mid:] > 0
    images[bad][:, x_mid:][region] = (60, 64, 68)
    res = reconstruct(images, capture.K, voxel_mm=2.0)
    assert res.cameras[bad] is None
    assert "不一致" in res.rejected[bad]
