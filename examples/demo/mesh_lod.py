"""Lighter robot meshes for a web demo, without changing its collision geometry.

A scene built from mjlab carries every robot mesh at full CAD detail as STL, which is
most of the bytes a browser downloads. :func:`lighten_meshes` rewrites them in place on
the scene's ``MjSpec``:

- a mesh only drawn is decimated (quadric simplification), per a YAML config;
- a mesh that collides is replaced by its convex hull, the shape MuJoCo collides against
  anyway; one both drawn and colliding is split into the two;
- every body keeps the mass and inertia it compiled to before, so lighter visual meshes
  cannot shift the dynamics;
- every mesh is stored inline (shared vertices, rounded) instead of as STL, which spends
  50 bytes a triangle repeating each vertex.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fast_simplification
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial import ConvexHull


@dataclass
class MeshStats:
    meshes: int = 0
    faces_before: int = 0
    faces_after: int = 0
    stl_bytes: int = 0


def _load(data: bytes, name: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(
        io.BytesIO(data), file_type=Path(name).suffix[1:].lower(), process=True
    )
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(
        mesh.faces, dtype=np.int64
    )


def _decimate(vertices, faces, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(faces)
    if n <= settings["min_faces"]:
        return vertices, faces
    keep = int(round(n * (1.0 - settings["target_reduction"])))
    keep = int(np.clip(keep, settings["min_faces"], settings["max_faces"]))
    if n <= keep:
        return vertices, faces
    v, f = fast_simplification.simplify(
        vertices.astype(np.float32),
        faces.astype(np.int32),
        target_reduction=1.0 - keep / n,
        agg=settings["aggressiveness"],
    )
    return np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int64)


def _hull(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(vertices)
    used = np.unique(hull.simplices)
    remap = np.full(len(vertices), -1)
    remap[used] = np.arange(len(used))
    faces = remap[hull.simplices]
    v = vertices[used]
    # Wind every face outward, as qhull does not orient them.
    normals = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]])
    flip = np.einsum("ij,ij->i", normals, v[faces[:, 0]] - v.mean(0)) < 0
    faces[flip] = faces[flip][:, ::-1]
    return v, faces


#: Mesh settings carried over when a mesh is replaced by its inline version.
_KEPT = (
    "scale",
    "refpos",
    "refquat",
    "inertia",
    "smoothnormal",
    "maxhullvert",
    "material",
)


def _set_inline(
    spec: mujoco.MjSpec, mesh, vertices: np.ndarray, faces: np.ndarray, precision: float
):
    """Replace ``mesh`` by a fresh one of the same name holding the data inline.

    A fresh mesh, not the old one edited: a compiled spec keeps each mesh's processed
    faces and normals, which would no longer match new faces.
    """
    kept = {attr: getattr(mesh, attr) for attr in _KEPT}
    name = mesh.name
    spec.delete(mesh)
    mesh = spec.add_mesh(name=name)
    for attr, value in kept.items():
        setattr(mesh, attr, value)
    # Rounding can make vertices coincide: merge them here and drop the triangles that
    # collapse, since MuJoCo's own merge would leave its face normals out of step.
    rounded = np.round(vertices / precision).astype(np.int64)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = np.unique(faces[keep], axis=0)
    mesh.uservert = (unique * precision).ravel().tolist()
    mesh.userface = faces.ravel().tolist()
    return mesh


def _asset_key(spec: mujoco.MjSpec, file: str) -> str | None:
    for key in spec.assets:
        if key == file or key.endswith("/" + file):
            return key
    return None


def _freeze_inertia(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
    for body in spec.bodies:
        if not body.name or body.name == "world" or body.explicitinertial:
            continue
        b = model.body(body.name)
        body.mass = float(b.mass[0])
        body.ipos = b.ipos.copy()
        body.iquat = b.iquat.copy()
        body.inertia = b.inertia.copy()
        body.explicitinertial = True


def lighten_meshes(spec: mujoco.MjSpec, config_path: Path) -> MeshStats:
    """Decimate drawn meshes and hull colliding ones on ``spec``, in place."""
    config = yaml.safe_load(Path(config_path).read_text())
    defaults = {
        k: config[k]
        for k in ("aggressiveness", "target_reduction", "min_faces", "max_faces")
    }
    overrides = config.get("overrides") or {}
    precision = float(config["vertex_precision"])

    model = spec.compile()
    # Which way each mesh is used, and the spec geoms that collide with it.
    spec_geoms = list(spec.geoms)
    if len(spec_geoms) != model.ngeom:
        raise RuntimeError("spec and model geoms do not line up")
    drawn: set[str] = set()
    colliding: dict[str, list] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = model.mesh(model.geom_dataid[g]).name
        if spec_geoms[g].meshname != name:
            raise RuntimeError(
                f"geom {g} names mesh {spec_geoms[g].meshname!r}, model {name!r}"
            )
        if model.geom_contype[g] or model.geom_conaffinity[g]:
            colliding.setdefault(name, []).append(spec_geoms[g])
        else:
            drawn.add(name)

    _freeze_inertia(spec, model)

    stats = MeshStats()
    used_keys: dict[str, int] = {}
    for mesh in spec.meshes:
        if mesh.file:
            key = _asset_key(spec, mesh.file)
            if key is not None:
                used_keys[key] = used_keys.get(key, 0) + 1

    for mesh in list(spec.meshes):
        if not mesh.file:
            continue  # inline meshes (the logo) are already light
        key = _asset_key(spec, mesh.file)
        if key is None:
            raise RuntimeError(f"no asset holds {mesh.file!r}")
        data = spec.assets[key]
        vertices, faces = _load(data, mesh.file)
        stats.meshes += 1
        stats.faces_before += len(faces)
        stats.stl_bytes += len(data)
        file = mesh.file

        if mesh.name in drawn:
            settings = {**defaults, **overrides.get(file, {})}
            v, f = _decimate(vertices, faces, settings)
            if mesh.name in colliding:
                hv, hf = _hull(vertices)
                hull = spec.add_mesh(name=f"{mesh.name}_hull")
                hull.scale = mesh.scale
                hull = _set_inline(spec, hull, hv, hf, precision)
                for geom in colliding[mesh.name]:
                    geom.meshname = hull.name
                stats.faces_after += len(hf)
        elif mesh.name in colliding:
            v, f = _hull(vertices)
        else:
            v, f = vertices, faces  # unused by any geom: kept, inline
        _set_inline(spec, mesh, v, f, precision)
        stats.faces_after += len(f)

        used_keys[key] -= 1
        if used_keys[key] == 0:
            del spec.assets[key]
    return stats
