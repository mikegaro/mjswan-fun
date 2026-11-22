"""LIRA logo drop

The LIRA laboratory logo in 3D, after the mujoco.org landing page: the gold letters
stand on the floor and the navy gear falls onto them. There is no robot and no policy,
only physics; drag the gear around, or reset to drop it again.

The geometry is traced from the logo image at build time, so nothing is vendored. Point
``LIRA_LOGO`` at another copy of the PNG to use it instead of the default path.

Each shape (L, I, R, A and the gear) is cut into thin horizontal bands, one convex
trapezoid per filled run of a band. Trapezoids that stay on one straight edge are merged:
tightly for the visual mesh, which follows the outline, holes included, and loosely for
the collision pieces, each extruded into one convex mesh.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy import ndimage

import mjswan

LOGO_PATH = Path(os.getenv("LIRA_LOGO", "/home/miguel/website_v1/images/logo.png"))

CAP_HEIGHT = 1.0  # metres from the letters' baseline to their top
DEPTH = 0.35  # extrusion depth of every shape, metres
UPSCALE = 4  # supersampling of the traced masks, which smooths the 512 px edges
VISUAL_TOL = 0.002  # metres a merged visual trapezoid may stray from the outline
COLLISION_TOL = 0.03  # the same, for a collision piece
GEAR_LIFT = (
    1.2  # metres between the letter tops and the gear's lowest point at the start
)
GEAR_TILT = (2.0, 6.0, 4.0)  # degrees about x, y, z, so the gear tumbles as it falls
GEAR_DENSITY = 300.0  # kg/m^3
FRICTION = (1.0, 0.02, 0.001)
SPOT_HEIGHT = 5.0  # metres above the floor
SPOT_CUTOFF = 35.0  # spotlight half-angle, degrees


@dataclass
class Shape:
    """One traced shape: its trapezoids in metres, logo plane (x right, z up)."""

    name: str
    rgba: tuple[float, float, float, float]
    visual: list[tuple[float, ...]]
    collision: list[tuple[float, ...]]


# ── tracing ───────────────────────────────────────────────────


def _intervals(row: np.ndarray) -> list[tuple[float, float]]:
    """Filled runs of one field row (inside is > 0), ends found to a subpixel."""
    inside = row > 0
    edges = np.flatnonzero(inside[1:] != inside[:-1])
    xs = [j + row[j] / (row[j] - row[j + 1]) for j in edges]
    return list(zip(xs[0::2], xs[1::2]))


def _overlapping(iv: tuple[float, float], ivs: list[tuple[float, float]]) -> list[int]:
    return [k for k, (a, b) in enumerate(ivs) if a < iv[1] and iv[0] < b]


def _band_trapezoids(field: np.ndarray):
    """Thin trapezoids between consecutive rows, with the links that chain them.

    A trapezoid is ``(y0, xl0, xr0, y1, xl1, xr1)`` in field pixels, ``y`` down. Its ends
    are runs of the two rows it spans when the run maps one to one through the band's
    middle; where the shape splits or joins inside a band it is the middle run, square.
    """
    rows = [_intervals(r) for r in field]
    traps, top_key, bottom_key = [], [], []
    for r in range(len(field) - 1):
        top, bottom = rows[r], rows[r + 1]
        mid = _intervals((field[r] + field[r + 1]) / 2)
        for iv in mid:
            t, b = _overlapping(iv, top), _overlapping(iv, bottom)
            t_ok = len(t) == 1 and len(_overlapping(top[t[0]], mid)) == 1
            b_ok = len(b) == 1 and len(_overlapping(bottom[b[0]], mid)) == 1
            if t_ok and b_ok:
                traps.append((r, *top[t[0]], r + 1, *bottom[b[0]]))
                top_key.append((r, t[0]))
                bottom_key.append((r + 1, b[0]))
            else:
                traps.append((r, *iv, r + 1, *iv))
                top_key.append(None)
                bottom_key.append(None)
    return traps, top_key, bottom_key


def _chains(traps, top_key, bottom_key) -> list[list[int]]:
    """Runs of trapezoids where each one's bottom is exactly the next one's top."""
    by_top = {k: i for i, k in enumerate(top_key) if k is not None}
    has_parent = set()
    nxt = {}
    for i, k in enumerate(bottom_key):
        if k is not None and k in by_top:
            nxt[i] = by_top[k]
            has_parent.add(by_top[k])
    chains = []
    for i in range(len(traps)):
        if i in has_parent:
            continue
        chain = [i]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        chains.append(chain)
    return chains


def _merge(traps, chain: list[int], tol: float) -> list[tuple[float, ...]]:
    """Greedily merge a chain into trapezoids whose sides stay within ``tol`` pixels."""
    ys = [traps[chain[0]][0]] + [traps[i][3] for i in chain]
    xl = [traps[chain[0]][1]] + [traps[i][4] for i in chain]
    xr = [traps[chain[0]][2]] + [traps[i][5] for i in chain]
    out, i = [], 0
    while i < len(ys) - 1:
        j = i + 1
        while j + 1 < len(ys):
            k = np.arange(i + 1, j + 1)
            s = (np.array(ys)[k] - ys[i]) / (ys[j + 1] - ys[i])
            dl = np.abs(np.array(xl)[k] - (xl[i] + s * (xl[j + 1] - xl[i])))
            dr = np.abs(np.array(xr)[k] - (xr[i] + s * (xr[j + 1] - xr[i])))
            if max(dl.max(), dr.max()) > tol:
                break
            j += 1
        out.append((ys[i], xl[i], xr[i], ys[j], xl[j], xr[j]))
        i = j
    return out


def trace_logo(path: Path) -> list[Shape]:
    """The logo's five shapes, gold letters first (left to right), then the gear."""
    image = np.asarray(Image.open(path).convert("RGBA")).astype(float)
    rgb, opaque = image[..., :3], image[..., 3] > 128

    # Two colours: k-means seeded with the bluest and the reddest opaque pixel.
    colours = rgb[opaque]
    centres = np.array(
        [colours[colours[:, 2].argmax()], colours[colours[:, 0].argmax()]]
    )
    for _ in range(10):
        label = ((colours[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([colours[label == k].mean(0) for k in range(2)])
    label = ((rgb[..., None, :] - centres[None, None]) ** 2).sum(-1).argmin(-1)
    navy, gold = centres

    components = []
    for colour, mask in ((gold, opaque & (label == 1)), (navy, opaque & (label == 0))):
        labels, n = ndimage.label(mask)
        for k, sl in enumerate(ndimage.find_objects(labels), start=1):
            area = (labels[sl] == k).sum()
            if area > 200:  # drop anti-aliasing specks
                components.append((colour, labels == k, sl))

    letters = sorted(
        (c for c in components if c[0] is gold), key=lambda c: c[2][1].start
    )
    gears = [c for c in components if c[0] is navy]
    if len(letters) != 4 or len(gears) != 1:
        raise ValueError(
            f"expected 4 letters and 1 gear in {path}, got {len(letters)} and {len(gears)}"
        )

    baseline = max(c[2][0].stop for c in letters)
    cap = baseline - min(c[2][0].start for c in letters)
    metres_per_px = CAP_HEIGHT / cap

    shapes = []
    for name, (colour, mask, sl) in zip(
        ["L", "I", "R", "A", "gear"], [*letters, *gears]
    ):
        pad = 3
        y0, x0 = sl[0].start - pad, sl[1].start - pad
        crop = np.pad(mask, pad)[
            sl[0].start : sl[0].stop + 2 * pad, sl[1].start : sl[1].stop + 2 * pad
        ]
        h, w = crop.shape
        big = Image.fromarray((crop * 255).astype(np.uint8)).resize(
            (w * UPSCALE, h * UPSCALE), Image.Resampling.BILINEAR
        )
        field = (
            ndimage.gaussian_filter(np.asarray(big) / 255.0, sigma=0.6 * UPSCALE) - 0.5
        )

        traps, top_key, bottom_key = _band_trapezoids(field)
        chains = _chains(traps, top_key, bottom_key)

        def to_metres(t):
            # Field pixel centres back to image pixels, then to metres from the baseline.
            def px(v, origin):
                return (v + 0.5) / UPSCALE - 0.5 + origin

            ya, xla, xra, yb, xlb, xrb = t
            za = (baseline - px(ya, y0)) * metres_per_px
            zb = (baseline - px(yb, y0)) * metres_per_px
            return (
                za, px(xla, x0) * metres_per_px, px(xra, x0) * metres_per_px,
                zb, px(xlb, x0) * metres_per_px, px(xrb, x0) * metres_per_px,
            )  # fmt: skip

        tol_px = UPSCALE / metres_per_px  # field pixels per metre
        visual = [
            to_metres(t) for c in chains for t in _merge(traps, c, VISUAL_TOL * tol_px)
        ]
        collision = [
            to_metres(t)
            for c in chains
            for t in _merge(traps, c, COLLISION_TOL * tol_px)
        ]
        rgba = (*(colour / 255.0), 1.0)
        shapes.append(Shape(name, rgba, visual, collision))
    return shapes


# ── meshes ────────────────────────────────────────────────────


class _MeshBuilder:
    def __init__(self) -> None:
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, int, int]] = []

    def quad(self, a, b, c, d, outward) -> None:
        """Two triangles over the planar quad a-b-c-d, wound to face ``outward``."""
        n = np.cross(np.subtract(b, a), np.subtract(c, a))
        if np.dot(n, outward) < 0:
            a, b, c, d = d, c, b, a
        base = len(self.verts)
        self.verts += [tuple(a), tuple(b), tuple(c), tuple(d)]
        self.faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]


def _p(x, z, y):
    return (x, y, z)


def _union_at(traps, z, side) -> list[tuple[float, float]]:
    """Union of the x-runs at height ``z`` of the trapezoids just above or below it."""
    runs = []
    for za, xla, xra, zb, xlb, xrb in traps:  # za is the top (higher z)
        if (side > 0 and zb <= z < za) or (side < 0 and zb < z <= za):
            s = (za - z) / (za - zb)
            runs.append((xla + s * (xlb - xla), xra + s * (xrb - xra)))
    runs.sort()
    merged: list[list[float]] = []
    for a, b in runs:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(m) for m in merged]


def _difference(a, b) -> list[tuple[float, float]]:
    """Runs in ``a`` not covered by ``b`` (both sorted and disjoint)."""
    out = []
    for lo, hi in a:
        cur = lo
        for blo, bhi in b:
            if bhi <= cur or blo >= hi:
                continue
            if blo > cur:
                out.append((cur, blo))
            cur = max(cur, bhi)
        if cur < hi:
            out.append((cur, hi))
    return [r for r in out if r[1] - r[0] > 1e-6]


def visual_mesh(traps, origin) -> tuple[np.ndarray, np.ndarray]:
    """A closed extrusion of the trapezoids: faces, outer walls and the steps between."""
    ox, oz = origin
    yf, yb = -DEPTH / 2, DEPTH / 2  # front faces the viewer, at -y
    m = _MeshBuilder()
    for za, xla, xra, zb, xlb, xrb in traps:
        xla, xra, xlb, xrb = xla - ox, xra - ox, xlb - ox, xrb - ox
        za, zb = za - oz, zb - oz
        for y, out in ((yf, (0, -1, 0)), (yb, (0, 1, 0))):
            m.quad(_p(xla, za, y), _p(xra, za, y), _p(xrb, zb, y), _p(xlb, zb, y), out)
        m.quad(
            _p(xla, za, yf),
            _p(xla, za, yb),
            _p(xlb, zb, yb),
            _p(xlb, zb, yf),
            (-1, 0, 0),
        )
        m.quad(
            _p(xra, za, yf),
            _p(xra, za, yb),
            _p(xrb, zb, yb),
            _p(xrb, zb, yf),
            (1, 0, 0),
        )
    for z in sorted({t[0] for t in traps} | {t[3] for t in traps}):
        above, below = _union_at(traps, z, +1), _union_at(traps, z, -1)
        for runs, out in (
            (_difference(above, below), (0, 0, -1)),
            (_difference(below, above), (0, 0, 1)),
        ):
            for a, b in runs:
                zz = z - oz
                a, b = a - ox, b - ox
                m.quad(_p(a, zz, yf), _p(b, zz, yf), _p(b, zz, yb), _p(a, zz, yb), out)
    return np.array(m.verts, dtype=float), np.array(m.faces, dtype=int)


def prism(trap, origin) -> np.ndarray:
    """The eight corners of one trapezoid extruded through the depth."""
    ox, oz = origin
    za, xla, xra, zb, xlb, xrb = trap
    return np.array(
        [
            (x - ox, y, z - oz)
            for y in (-DEPTH / 2, DEPTH / 2)
            for x, z in ((xla, za), (xra, za), (xrb, zb), (xlb, zb))
        ]
    )


# ── scene ─────────────────────────────────────────────────────


def _euler_quat(deg: tuple[float, float, float]) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.radians(deg), "xyz")
    return quat


def build_spec(path: Path = LOGO_PATH) -> mujoco.MjSpec:
    shapes = trace_logo(path)
    letters, gear = shapes[:4], shapes[4]
    left = min(t[1] for s in letters for t in s.visual)
    right = max(t[2] for s in letters for t in s.visual)
    cx = (left + right) / 2  # centre the word on x = 0

    spec = mujoco.MjSpec()
    spec.modelname = "lira_logo"
    spec.option.timestep = 0.002
    spec.visual.headlight.ambient = [0.2, 0.2, 0.2]
    spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.stat.center = [0, 0, 1.2]
    spec.stat.extent = 4.0

    # MuJoCo's default blue scene, as in its example models: gradient sky, checker floor.
    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7], rgb2=[0.0, 0.0, 0.0], width=512, height=3072,
    )  # fmt: skip
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4], rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8], width=300, height=300,
    )  # fmt: skip
    floor_mat = spec.add_material(
        name="floor", texrepeat=[5, 5], texuniform=True, reflectance=0.2
    )
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    world = spec.worldbody
    # A spotlight straight above the word, tilted a little toward the viewer (-y).
    world.add_light(
        name="spot", pos=[0.0, -1.0, SPOT_HEIGHT], dir=[0.0, 0.2, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_SPOT, cutoff=SPOT_CUTOFF, exponent=8,
        castshadow=True, diffuse=[1.0, 1.0, 1.0], specular=[0.3, 0.3, 0.3],
    )  # fmt: skip
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="floor",
                   friction=FRICTION)  # fmt: skip

    def add_shape(parent, shape: Shape, origin, density: float) -> None:
        mat = spec.add_material(
            name=f"{shape.name}_mat", rgba=list(shape.rgba), specular=0.4, shininess=0.5
        )
        verts, faces = visual_mesh(shape.visual, origin)
        mesh = spec.add_mesh(name=f"{shape.name}_visual", uservert=verts.ravel().tolist(),
                             userface=faces.ravel().tolist())  # fmt: skip
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        parent.add_geom(name=f"{shape.name}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                        material=mat.name, contype=0, conaffinity=0, group=1, density=0)  # fmt: skip
        for i, trap in enumerate(shape.collision):
            piece = spec.add_mesh(
                name=f"{shape.name}_col{i}",
                uservert=prism(trap, origin).ravel().tolist(),
            )
            parent.add_geom(name=f"{shape.name}_col{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=piece.name,
                            group=3, rgba=[0.8, 0.3, 0.3, 0.5], density=density, friction=FRICTION)  # fmt: skip

    for shape in letters:
        add_shape(world, shape, (cx, 0.0), density=1000.0)

    # The gear's body sits at its outline's centre, so it tumbles about a sensible point.
    gx = float(np.mean([(t[1] + t[2] + t[4] + t[5]) / 4 for t in gear.visual]))
    gz = float(np.mean([(t[0] + t[3]) / 2 for t in gear.visual]))
    gear_low = min(t[3] for t in gear.visual)
    start_z = CAP_HEIGHT + GEAR_LIFT + (gz - gear_low)
    body = world.add_body(
        name="gear", pos=[gx - cx, 0, start_z], quat=_euler_quat(GEAR_TILT)
    )
    body.add_freejoint(name="gear_free")
    add_shape(body, gear, (gx, gz), density=GEAR_DENSITY)

    qpos = np.concatenate([[gx - cx, 0, start_z], _euler_quat(GEAR_TILT)])
    spec.add_key(name="home", qpos=qpos.tolist())
    return spec


def add_projects(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="LIRA")
    scene = project.add_scene(name="LIRA Logo", spec=build_spec())
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 1.0),
            distance=6.5,
            elevation=-12.0,
            azimuth=90.0,
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
        )
    )


def main() -> None:
    """Build and launch the demo.

    Environment variables (see ``lira_logo_website.md`` for the website build):
        LIRA_LOGO: The logo PNG to trace.
        MJSWAN_BASE_PATH: Base path for deployment (default: '/').
        MJSWAN_OUT_DIR: Where to build (default: the shared examples/demo/dist).
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser.
    """
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    add_projects(builder)
    out_dir = os.getenv("MJSWAN_OUT_DIR")
    app = builder.build(output_dir=Path(out_dir)) if out_dir else builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
