"""命令行入口。

    python -m foot_scanner3d demo  --out out/            # 合成一只脚，重建并和真值对比
    python -m foot_scanner3d bench --trials 5            # 多只随机脚的误差统计
    python -m foot_scanner3d scan frames/*.jpg --focal-px 1450 --out out/   # 真实帧（实验性）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import DEFAULT_EQUIV_FOCAL_MM, focal_from_equiv, intrinsics
from .measure import MEASUREMENT_LABELS, measure_mesh
from .pipeline import ScanResult, pose_errors, reconstruct
from .synth import make_capture
from .viz import debug_montage


def _save(res: ScanResult, images, out: Path, gt=None, extra=()):
    out.mkdir(parents=True, exist_ok=True)
    res.mesh.save_obj(out / "foot.obj")
    payload = {"measurements": res.measurements.as_dict(), "views_used": res.n_used, "rejected": res.rejected}
    if gt is not None:
        payload["ground_truth"] = gt[0].as_dict()
    (out / "measurements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    # 调试图用一张低仰角、纸被部分遮挡的帧
    idx = next((i for i, f in enumerate(res.fits) if f is not None), 0)
    img = debug_montage(images[idx], res.masks[idx], res.fits[idx], res.mesh,
                        (res.measurements, res.detail), gt, list(extra))
    cv2.imwrite(str(out / "debug.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def _table(rec, gt=None) -> str:
    r = rec.as_dict()
    g = gt.as_dict() if gt is not None else None
    lines = []
    for k, label in MEASUREMENT_LABELS.items():
        s = f"{label:<6}\t{r[k]:7.1f} mm"
        if g is not None:
            s += f"\t真值 {g[k]:7.1f}\t误差 {r[k] - g[k]:+5.1f}"
        lines.append(s)
    lines.append("足弓高度\t—（需要看到脚底，visual hull 无法测量）")
    return "\n".join(lines)


def cmd_demo(args):
    t0 = time.time()
    cap = make_capture(seed=args.seed, n_views=args.views)
    t1 = time.time()
    res = reconstruct(cap.images, cap.K, voxel_mm=args.voxel)
    t2 = time.time()
    gt = measure_mesh(cap.gt_mesh)
    pe = pose_errors(res.cameras, cap.cameras)
    print(_table(res.measurements, gt[0]))
    print(f"\n可用帧 {res.n_used}/{len(cap.images)}；相机位置误差 均值 {pe['center_mean_mm']:.2f} mm / 最大 {pe['center_max_mm']:.2f} mm")
    print(f"合成 {t1 - t0:.0f} s，重建+测量 {t2 - t1:.0f} s")
    extra = [f"views used {res.n_used}/{len(cap.images)}, voxel {args.voxel} mm",
             f"camera pose error mean {pe['center_mean_mm']:.2f} mm, max {pe['center_max_mm']:.2f} mm"]
    _save(res, cap.images, Path(args.out), gt, extra)
    print(f"输出：{args.out}/foot.obj, measurements.json, debug.jpg")


def cmd_bench(args):
    rows = []
    for k in range(args.trials):
        seed = args.seed + k
        cap = make_capture(seed=seed, n_views=args.views)
        res = reconstruct(cap.images, cap.K, voxel_mm=args.voxel)
        gt = measure_mesh(cap.gt_mesh)[0].as_dict()
        r = res.measurements.as_dict()
        err = {key: r[key] - gt[key] for key in MEASUREMENT_LABELS}
        rows.append(err)
        print(f"seed {seed}: " + "  ".join(f"{MEASUREMENT_LABELS[key]} {e:+.1f}" for key, e in err.items()), flush=True)
    print("\n项目\tMAE\t最大绝对误差 (mm)")
    for key, label in MEASUREMENT_LABELS.items():
        e = np.abs([row[key] for row in rows])
        print(f"{label}\t{e.mean():.2f}\t{e.max():.2f}")


def cmd_scan(args):
    paths = sorted(args.images) if args.sort else args.images
    images = [cv2.imread(str(p)) for p in paths]
    if any(im is None for im in images):
        raise SystemExit("有图片读不出来")
    h, w = images[0].shape[:2]
    f = args.focal_px or focal_from_equiv(args.equiv_focal or DEFAULT_EQUIV_FOCAL_MM, (w, h))
    res = reconstruct(images, intrinsics(f, (w, h)), voxel_mm=args.voxel)
    print(_table(res.measurements))
    print(f"\n可用帧 {res.n_used}/{len(images)}")
    for i, why in res.rejected.items():
        print(f"  丢弃 {paths[i]}：{why}")
    _save(res, images, Path(args.out), extra=[f"views used {res.n_used}/{len(images)}"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="foot_scanner3d", description="多视角 3D 脚型重建原型")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="合成数据端到端演示")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--views", type=int, default=48)
    d.add_argument("--voxel", type=float, default=1.0)
    d.add_argument("--out", default="out")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("bench", help="多只随机脚的误差统计")
    b.add_argument("--trials", type=int, default=5)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--views", type=int, default=48)
    b.add_argument("--voxel", type=float, default=1.0)
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("scan", help="从真实照片重建（实验性：分割仍是颜色阈值）")
    s.add_argument("images", nargs="+", type=Path, help="按拍摄顺序排列的帧")
    s.add_argument("--focal-px", type=float, help="焦距（像素），ARKit 可直接给出")
    s.add_argument("--equiv-focal", type=float, help="35mm 等效焦距（EXIF）")
    s.add_argument("--sort", action="store_true", help="按文件名排序")
    s.add_argument("--voxel", type=float, default=1.0)
    s.add_argument("--out", default="out")
    s.set_defaults(func=cmd_scan)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
