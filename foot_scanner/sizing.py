"""脚长 -> 鞋码换算（参考值）。

运动鞋品牌普遍以"厘米码"（= 脚长，0.5 cm 一档）为基准，再对应到美码、欧码；
不同品牌、不同鞋款的楦型差异可达半码到一码，这里给的是常见运动鞋尺码表的
近似线性关系，只作推荐起点。
"""

from __future__ import annotations

import math


def _ceil_to(x: float, step: float) -> float:
    return math.ceil(x / step - 1e-9) * step


def recommend_sizes(foot_length_mm: float) -> dict:
    cm = _ceil_to(foot_length_mm / 10.0, 0.5)  # 向上取到半码，宁大勿小
    return {
        "foot_length_mm": round(foot_length_mm, 1),
        "mondopoint_mm": int(round(foot_length_mm)),  # ISO 9407 / 国标 GB/T 3293.1 以脚长毫米计
        "cm": cm,  # 厘米码 / JP 码
        "cn": int(_ceil_to(foot_length_mm, 5)),  # 国标鞋号，5 mm 一档
        "us_men": round(cm - 18.0, 1),  # 常见运动鞋表：26 cm ≈ US 8
        "us_women": round(cm - 17.0, 1),  # 常见运动鞋表：24 cm ≈ US 7
        "eu": round(_ceil_to((cm + 1.5) * 1.5, 0.5), 1),  # 巴黎码近似：(脚长 cm + 1.5) × 1.5
        "note": "参考值，不同品牌楦型可相差半码到一码",
    }
