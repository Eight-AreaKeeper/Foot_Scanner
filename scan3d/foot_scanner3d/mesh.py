"""三角网格的最小工具集：剖面、凸包周长、导出。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes


@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) 毫米
    faces: np.ndarray  # (F, 3) 顶点索引，法向朝外

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    def face_normals(self) -> np.ndarray:
        tri = self.triangles
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def volume(self) -> float:
        tri = self.triangles
        return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    def transformed(self, rot: np.ndarray, trans: np.ndarray) -> "Mesh":
        return Mesh(self.vertices @ rot.T + trans, self.faces.copy())

    def cropped_z(self, z_max: float) -> "Mesh":
        """只保留所有顶点都低于 z_max 的面（用于和重建范围对齐）。"""
        keep = (self.vertices[self.faces][:, :, 2] <= z_max).all(axis=1)
        return Mesh(self.vertices, self.faces[keep])

    def save_obj(self, path: str | Path) -> None:
        with open(path, "w") as fh:
            fh.write("# foot_scanner3d mesh, units: mm\n")
            np.savetxt(fh, self.vertices, fmt="v %.3f %.3f %.3f")
            np.savetxt(fh, self.faces + 1, fmt="f %d %d %d")


def mesh_from_field(field: np.ndarray, origin: np.ndarray, spacing: float) -> Mesh:
    """field 在物体内部为正、外部为负；返回 0 等值面。"""
    verts, faces, _, _ = marching_cubes(field, level=0.0, spacing=(spacing,) * 3)
    mesh = Mesh(verts + origin, faces.astype(np.int64))
    if mesh.volume() < 0:
        mesh.faces = mesh.faces[:, ::-1].copy()
    return mesh


def section_points(mesh: Mesh, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """网格与平面的交线，返回交点 (M, 3)。"""
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    d = (mesh.vertices - point) @ normal
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        ia, ib = mesh.faces[:, a], mesh.faces[:, b]
        da, db = d[ia], d[ib]
        hit = (da > 0) != (db > 0)
        if not hit.any():
            continue
        w = (da[hit] / (da[hit] - db[hit]))[:, None]
        out.append(mesh.vertices[ia[hit]] + w * (mesh.vertices[ib[hit]] - mesh.vertices[ia[hit]]))
    return np.concatenate(out) if out else np.zeros((0, 3))


def hull_perimeter(points_2d: np.ndarray) -> float:
    """二维凸包周长：皮尺绕一圈量到的就是凸包，凹处会被跨过去。"""
    if len(points_2d) < 3:
        return 0.0
    return float(ConvexHull(points_2d).area)  # 2D 时 area 是周长
