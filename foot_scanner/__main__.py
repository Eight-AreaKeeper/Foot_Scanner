"""命令行：python -m foot_scanner photo.jpg [--debug out.png]"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from .paper import PAPER_SIZES_MM, PaperNotFound
from .pipeline import read_equiv_focal, render_debug, scan
from .segment import FootNotFound


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="foot_scanner", description="用一张俯拍照片和一张参照纸测量脚长、脚宽")
    p.add_argument("image", help="俯拍照片：脚踩在参照纸上，纸的四个角都在画面内")
    p.add_argument("--paper", default="a4", choices=sorted(PAPER_SIZES_MM), help="参照纸尺寸（默认 a4）")
    p.add_argument("--mode", default="free", choices=["free", "wall"], help="free：脚在纸中间；wall：纸贴墙、脚跟贴墙")
    p.add_argument("--focal35", type=float, help="35 mm 等效焦距；不填则读 EXIF，再不行就从图像估计")
    p.add_argument("--no-parallax", action="store_true", help="关闭视差校正")
    p.add_argument("--debug", help="输出带测量线的调试图")
    args = p.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图片: {args.image}", file=sys.stderr)
        return 2
    focal = args.focal35 or read_equiv_focal(args.image)
    try:
        result = scan(image, args.paper, args.mode, focal, parallax=not args.no_parallax)
    except (PaperNotFound, FootNotFound, ValueError) as e:
        print(f"测量失败: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.debug:
        cv2.imwrite(args.debug, render_debug(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
