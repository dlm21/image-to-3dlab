#!/usr/bin/env python3
"""Rebuild a hollow, holed shell as one solid, watertight skin, and re-bake its colour onto it.

    python scripts/solidify_mesh.py in.glb out.glb [--faces 300000] [--resolution 512]
        [--close 8] [--texture-size 2048] [--report report.json]

**The problem.** TRELLIS.2's remesh (`to_glb(remesh=True)`) contours the *unsigned*
distance to the decoded surface, one voxel out. That wraps every surface in two skins: an
outer one facing out and an inner one a voxel below it facing in. Wherever the decode had
a hole, the two skins join round its rim and leave a tunnel. Measured on a 296k-face
boulder: a shell 0.2% of the rock's size thick, enclosing 0.005 of volume against a
convex hull of 0.32, with ~90 tunnels through it. Rays through its centre cross four
surfaces where a solid crosses two. Zero open edges, so hole-fillers find nothing to fill;
the holes are tunnels, not gaps. Half the faces are the inner skin nobody ever sees.

**What this does.**

1. Samples the surface densely and marks the voxels it passes through.
2. Grows that mask by `--close` voxels, so any hole narrower than about twice that seals.
3. Flood-fills the outside from the grid's corner. Everything the flood cannot reach is
   solid, cavity included. Then shrinks it back by the same amount (a morphological
   "closing": grow, fill, shrink).
4. Builds a signed distance field whose sign comes from that solid but whose size is the
   *exact* distance to the original surface. The new skin lands on the old one, not on
   voxel stair-steps. Patched holes, which no surface reaches, fall back to the mask.
5. Marching cubes, decimate to `--faces`, keep the largest piece, light Taubin smoothing
   (a smoothing that does not shrink the mesh).
6. Unwraps the new mesh with xatlas and re-bakes the colour: every texel finds its nearest
   point on the original, reads the original's UVs there, and samples the original
   texture. A metallic-roughness map that is one flat value (TRELLIS's usually is) is
   replaced by plain material numbers; a varied one is re-baked like the colour.

**What it costs.** Detail finer than a voxel is resampled away: at 512 that is 0.2% of
the asset's size, below what TRELLIS's own 1024 grid resolves after its remesh. Holes
wider than about `2 * close` voxels stay open, because closing them would also fill real
concavities. The colour is resampled twice (TRELLIS's bake, then this one), which softens
it slightly. Thin parts thinner than a voxel (leaves, fur cards) become blobs: this is for
solid things, rocks and props and bodies, not foliage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from mesh_health import measure  # noqa: E402, RUF100

#: A metallic-roughness channel whose standard deviation is under this (in 0-1 units) is
#: one flat value, and a texture for it is wasted bytes. TRELLIS's boulder measured 0.009
#: for roughness and 0.002 for metalness.
FLAT_CHANNEL_STD = 0.02


# ----------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    origin: np.ndarray  # world position of voxel (0, 0, 0)'s centre
    pitch: float        # voxel edge length in world units
    shape: tuple[int, int, int]

    def to_world(self, index: np.ndarray) -> np.ndarray:
        return np.asarray(index, dtype=np.float64) * self.pitch + self.origin

    def to_index(self, points: np.ndarray) -> np.ndarray:
        return np.rint((points - self.origin) / self.pitch).astype(np.int64)


def grid_for(bounds: np.ndarray, resolution: int, pad: int) -> Grid:
    """A cubic-voxel grid with `resolution` voxels across the longest side, plus `pad`
    empty voxels all round, so growing the mask never reaches the edge and the flood fill
    always has an outside to start from."""
    lo, hi = np.asarray(bounds, dtype=np.float64)
    pitch = float((hi - lo).max()) / resolution
    shape = tuple(int(n) for n in np.ceil((hi - lo) / pitch).astype(int) + 2 * pad + 1)
    return Grid(origin=lo - pad * pitch, pitch=pitch, shape=shape)


def closest_on_triangles(points: np.ndarray, a: np.ndarray, b: np.ndarray,
                         c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closest point on each triangle (a, b, c) to each point, row by row.

    Returns (closest points, barycentric weights). Ericson's region test, vectorised;
    later assignments win, so they run in reverse of his early-return order.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = points - a, points - b, points - c
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2

    def safe(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        return np.where(np.isfinite(out), out, 0.0)

    n = len(points)
    bary = np.empty((n, 3))
    denom = va + vb + vc
    v, w = safe(vb, denom), safe(vc, denom)
    bary[:] = np.stack([1 - v - w, v, w], axis=1)

    def put(mask, weights):
        bary[mask] = weights[mask] if weights.ndim == 2 else weights

    t = safe(d4 - d3, (d4 - d3) + (d5 - d6))
    put((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), np.stack([0 * t, 1 - t, t], axis=1))
    t = safe(d2, d2 - d6)
    put((vb <= 0) & (d2 >= 0) & (d6 <= 0), np.stack([1 - t, 0 * t, t], axis=1))
    put((d6 >= 0) & (d5 <= d6), np.array([0.0, 0.0, 1.0]))
    t = safe(d1, d1 - d3)
    put((vc <= 0) & (d1 >= 0) & (d3 <= 0), np.stack([1 - t, t, 0 * t], axis=1))
    put((d3 >= 0) & (d4 <= d3), np.array([0.0, 1.0, 0.0]))
    put((d1 <= 0) & (d2 <= 0), np.array([1.0, 0.0, 0.0]))
    closest = a * bary[:, :1] + b * bary[:, 1:2] + c * bary[:, 2:]
    return closest, bary


class SurfaceIndex:
    """Nearest-point queries against a mesh, fast enough for millions of points.

    A KD-tree over dense surface samples finds a handful of candidate faces; the exact
    closest point is then computed on each candidate. trimesh's own `closest_point`
    is exact, but minutes slower at this scale.

    Not quite exact: random samples leave gaps, so now and then none of the candidates
    is the true nearest face. Measured near the surface with faces ~7x the sample
    spacing (TRELLIS's ratio): 99.6% exact at k=8, the rest within a sixth of a spacing,
    which is a twelfth of a voxel. k=4 was 98% and half a spacing.
    """

    def __init__(self, mesh: trimesh.Trimesh, spacing: float, seed: int = 0):
        self.triangles = np.asarray(mesh.triangles, dtype=np.float64)
        count = int(np.ceil(mesh.area / spacing ** 2))
        points, faces = trimesh.sample.sample_surface(mesh, count, seed=seed)
        # Centroids too, so a face too small to catch a random sample is still findable.
        self.points = np.vstack([points, self.triangles.mean(axis=1)])
        self.faces = np.concatenate([faces, np.arange(len(self.triangles))])
        self.tree = cKDTree(self.points)

    def closest(self, queries: np.ndarray, k: int = 8, chunk: int = 250_000
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(distance, face index, barycentric weights) of the nearest surface point."""
        n = len(queries)
        dist, face, bary = np.empty(n), np.empty(n, np.int64), np.empty((n, 3))
        for start in range(0, n, chunk):
            q = queries[start:start + chunk]
            _, nearest = self.tree.query(q, k=k, workers=-1)
            candidates = self.faces[nearest.reshape(len(q), -1)]
            best_d = np.full(len(q), np.inf)
            best_f = np.zeros(len(q), np.int64)
            best_b = np.zeros((len(q), 3))
            for j in range(candidates.shape[1]):
                tri = self.triangles[candidates[:, j]]
                point, weights = closest_on_triangles(q, tri[:, 0], tri[:, 1], tri[:, 2])
                d = np.linalg.norm(point - q, axis=1)
                better = d < best_d
                best_d[better], best_f[better] = d[better], candidates[better, j]
                best_b[better] = weights[better]
            dist[start:start + len(q)] = best_d
            face[start:start + len(q)] = best_f
            bary[start:start + len(q)] = best_b
        return dist, face, bary


def surface_mask(points: np.ndarray, grid: Grid) -> np.ndarray:
    """Voxels the sampled surface passes through."""
    mask = np.zeros(grid.shape, dtype=bool)
    index = np.clip(grid.to_index(points), 0, np.array(grid.shape) - 1)
    mask[tuple(index.T)] = True
    return mask


def solid_mask(surface: np.ndarray, close: int) -> np.ndarray:
    """Grow by `close`, flood-fill the outside, call the rest solid, shrink back.

    The flood starts at voxel (0, 0, 0), which `grid_for`'s padding keeps empty. Growing
    seals holes up to about `2 * close` voxels wide before the flood can pour through them.
    """
    cross = ndimage.generate_binary_structure(3, 1)
    grown = ndimage.binary_dilation(surface, cross, iterations=close) if close else surface
    labels, _ = ndimage.label(~grown, structure=cross)
    if labels[0, 0, 0] == 0:
        raise ValueError("grid corner is not empty: pad the grid by more than close")
    solid = labels != labels[0, 0, 0]
    if close:
        solid = ndimage.binary_erosion(solid, cross, iterations=close, border_value=0)
    return solid


def signed_field(solid: np.ndarray, grid: Grid, index: SurfaceIndex,
                 band: int = 3) -> np.ndarray:
    """Negative inside, positive outside; near the boundary, the true distance to the
    original surface, in voxels. Far from it, a flat +/-band.

    Where the solid's boundary was made by `solid_mask` patching a hole, no original
    surface is nearby; there the field is a flat +/-0.5 so the skin sits on the mask.
    """
    cross = ndimage.generate_binary_structure(3, 1)
    boundary = solid ^ ndimage.binary_erosion(solid, cross, border_value=0)
    near = ndimage.binary_dilation(boundary, cross, iterations=band)
    field = np.where(solid, -float(band), float(band)).astype(np.float32)
    cells = np.argwhere(near)
    distance, _, _ = index.closest(grid.to_world(cells))
    distance /= grid.pitch
    sign = np.where(solid[tuple(cells.T)], -1.0, 1.0)
    value = np.where(distance < 1.5, sign * np.minimum(distance, band), sign * 0.5)
    field[tuple(cells.T)] = value
    return field


def extract_surface(field: np.ndarray, grid: Grid) -> trimesh.Trimesh:
    """Marching cubes at zero, oriented so the volume is positive."""
    from skimage import measure as skmeasure

    vertices, faces, _, _ = skmeasure.marching_cubes(field, 0.0)
    mesh = trimesh.Trimesh(grid.to_world(vertices), faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def decimate(mesh: trimesh.Trimesh, faces: int, smooth: int = 5) -> trimesh.Trimesh:
    """Simplify to about `faces`, keep the largest piece, Taubin-smooth lightly.

    The largest-piece step matters: decimation occasionally pinches off a two-triangle
    sliver, which is a second component and a non-manifold edge for no benefit.
    """
    import fast_simplification

    if faces < len(mesh.faces):
        v, f = fast_simplification.simplify(
            mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
            1.0 - faces / len(mesh.faces))
        mesh = trimesh.Trimesh(v, f, process=True)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda part: len(part.faces))
    if smooth:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def solidify_geometry(source: trimesh.Trimesh, *, faces: int, resolution: int = 512,
                      close: int = 8, smooth: int = 5, index: SurfaceIndex | None = None,
                      log=print) -> trimesh.Trimesh:
    """Steps 1-5 of the module docstring: any surface in, one closed skin out."""
    grid = grid_for(source.bounds, resolution, pad=close + 4)
    if index is None:
        index = SurfaceIndex(source, spacing=grid.pitch * 0.5)
    started = time.time()
    solid = solid_mask(surface_mask(index.points, grid), close)
    log(f"  solid mask {grid.shape} in {time.time() - started:.1f}s")
    started = time.time()
    field = signed_field(solid, grid, index)
    del solid
    mesh = extract_surface(field, grid)
    del field
    log(f"  marching cubes: {len(mesh.faces):,} faces in {time.time() - started:.1f}s")
    return decimate(mesh, faces, smooth)


# ----------------------------------------------------------------------------------------
# Texture
# ----------------------------------------------------------------------------------------
def unwrap(mesh: trimesh.Trimesh, texture_size: int) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray]:
    """xatlas: (vertex map into mesh.vertices, faces, uv). UVs follow trimesh's
    convention (v up), which trimesh flips to glTF's on export."""
    import xatlas

    atlas = xatlas.Atlas()
    atlas.add_mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = texture_size
    pack.padding = 2
    pack.bilinear = True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmap, faces, uv = atlas[0]
    return vmap.astype(np.int64), faces.astype(np.int64), uv.astype(np.float64)


def uv_to_pixel(uv: np.ndarray, size: int) -> np.ndarray:
    """UV (v up) to continuous pixel coordinates (x right, y down, texel centres on
    integers). The one place this convention lives: the rasteriser and the sampler
    both use it, so a bake reads exactly where it writes."""
    return np.stack([uv[..., 0] * size - 0.5, (1.0 - uv[..., 1]) * size - 0.5], axis=-1)


def rasterize_uv(uv: np.ndarray, faces: np.ndarray, size: int, budget: int = 4_000_000
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every texel whose centre lies in a UV triangle: (rows, cols, face, barycentric).

    Triangles are bucketed by bounding-box size and tested in vectorised batches. A
    texel on an edge shared by two triangles is kept once.
    """
    tri = uv_to_pixel(uv, size)[faces]  # (F, 3, 2)
    lo = np.clip(np.floor(tri.min(axis=1)).astype(np.int64), 0, size - 1)
    hi = np.clip(np.ceil(tri.max(axis=1)).astype(np.int64), 0, size - 1)
    dims = hi - lo + 1
    order = np.argsort(dims[:, 0] * dims[:, 1], kind="stable")
    a, e0, e1 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    det = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    rows, cols, face_out, bary_out = [], [], [], []
    start = 0
    while start < len(order):
        width, height = dims[order[start]]
        end = start + 1
        # Grow the batch while the largest box in it times its length fits the budget.
        while end < len(order):
            w, h = np.maximum(dims[order[end]], (width, height))
            if w * h * (end - start + 1) > budget:
                break
            width, height, end = w, h, end + 1
        batch = order[start:end]
        start = end
        gx, gy = np.meshgrid(np.arange(width), np.arange(height))
        x = lo[batch, 0, None] + gx.ravel()[None, :]
        y = lo[batch, 1, None] + gy.ravel()[None, :]
        px, py = x - a[batch, 0, None], y - a[batch, 1, None]
        d = det[batch, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (px * e1[batch, 1, None] - py * e1[batch, 0, None]) / d
            w = (e0[batch, 0, None] * py - e0[batch, 1, None] * px) / d
            eps = 1e-7
            inside = ((v >= -eps) & (w >= -eps) & (v + w <= 1 + eps)
                      & (x <= hi[batch, 0, None]) & (y <= hi[batch, 1, None]) & (d != 0))
        which, cell = np.nonzero(inside)
        rows.append(y[which, cell])
        cols.append(x[which, cell])
        face_out.append(batch[which])
        vv, ww = v[which, cell], w[which, cell]
        bary_out.append(np.stack([1 - vv - ww, vv, ww], axis=1))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    face_out, bary = np.concatenate(face_out), np.concatenate(bary_out)
    _, first = np.unique(rows * size + cols, return_index=True)
    bary = np.clip(bary[first], 0.0, 1.0)
    bary /= bary.sum(axis=1, keepdims=True)
    return rows[first], cols[first], face_out[first], bary


def sample_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, C) image at UVs, clamped at the edges."""
    height, width = image.shape[:2]
    if height != width:
        pixel = np.stack([uv[:, 0] * width - 0.5, (1.0 - uv[:, 1]) * height - 0.5], axis=1)
    else:
        pixel = uv_to_pixel(uv, width)
    x = np.clip(pixel[:, 0], 0, width - 1)
    y = np.clip(pixel[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def fill_gutters(image: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Copy each empty texel from its nearest filled one, so bilinear filtering and
    mipmaps at island edges never pull in black."""
    if filled.all() or not filled.any():
        return image
    _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
    return image[iy, ix]


def flat_value(image: np.ndarray, channels: tuple[int, ...],
               tolerance: float = FLAT_CHANNEL_STD) -> tuple[float, ...] | None:
    """Mean of each channel (0-1) if every one is flat, else None."""
    values = image[..., list(channels)].reshape(-1, len(channels)).astype(np.float64) / 255.0
    if (values.std(axis=0) > tolerance).any():
        return None
    return tuple(float(x) for x in values.mean(axis=0))


def textured_source(scene_or_mesh) -> trimesh.Trimesh:
    """The one textured mesh in a loaded GLB. A multi-material scene is refused rather
    than flattened, because flattening it would silently drop all but one texture."""
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    geometry = [g for g in scene_or_mesh.dump() if isinstance(g, trimesh.Trimesh)]
    if len(geometry) != 1:
        raise ValueError(f"expected one mesh in the GLB, found {len(geometry)}")
    return geometry[0]


def bake_material(target: trimesh.Trimesh, source: trimesh.Trimesh, index: SurfaceIndex,
                  texture_size: int, log=print) -> tuple[trimesh.Trimesh, dict]:
    """Unwrap `target` and transfer `source`'s textures onto it (step 6)."""
    from PIL import Image

    material = getattr(source.visual, "material", None)
    uv_source = getattr(source.visual, "uv", None)
    base = getattr(material, "baseColorTexture", None)
    if material is None or uv_source is None or base is None:
        log("  source has no colour texture: geometry only")
        return target, {"baked": False}

    started = time.time()
    vmap, faces, uv = unwrap(target, texture_size)
    vertices = target.vertices[vmap]
    log(f"  unwrap: {len(faces):,} faces in {time.time() - started:.1f}s")
    started = time.time()
    rows, cols, face, bary = rasterize_uv(uv, faces, texture_size)
    points = np.einsum("nij,ni->nj", vertices[faces[face]], bary)
    _, source_face, source_bary = index.closest(points)
    source_uv = np.einsum("nij,ni->nj", uv_source[source.faces[source_face]], source_bary)

    def bake(image) -> np.ndarray:
        pixels = np.asarray(image.convert("RGBA"), dtype=np.float32)
        out = np.zeros((texture_size, texture_size, 4), dtype=np.float32)
        out[rows, cols] = sample_bilinear(pixels, source_uv)
        filled = np.zeros((texture_size, texture_size), dtype=bool)
        filled[rows, cols] = True
        return np.clip(fill_gutters(out, filled), 0, 255).astype(np.uint8)

    colour = bake(base)
    colour[..., 3] = 255  # TRELLIS's bake is opaque; a stray alpha would read as holes
    record = {"baked": True, "texture_size": texture_size,
              "coverage": round(len(rows) / texture_size ** 2, 4)}
    kwargs = {"baseColorTexture": Image.fromarray(colour[..., :3], "RGB"),
              "baseColorFactor": getattr(material, "baseColorFactor", None),
              "doubleSided": False, "alphaMode": "OPAQUE"}
    rough = getattr(material, "roughnessFactor", None)
    metal = getattr(material, "metallicFactor", None)
    rough = 1.0 if rough is None else float(rough)
    metal = 1.0 if metal is None else float(metal)
    mr = getattr(material, "metallicRoughnessTexture", None)
    if mr is not None:
        mr_pixels = np.asarray(mr.convert("RGB"))
        flat = flat_value(mr_pixels, (1, 2))  # glTF: G roughness, B metalness
        if flat is not None:
            rough, metal = rough * flat[0], metal * flat[1]
            record["metallic_roughness"] = "flat, replaced by factors"
        else:
            kwargs["metallicRoughnessTexture"] = Image.fromarray(bake(mr)[..., :3], "RGB")
            record["metallic_roughness"] = "re-baked"
    kwargs["roughnessFactor"], kwargs["metallicFactor"] = rough, metal
    record["roughness_factor"], record["metallic_factor"] = round(rough, 4), round(metal, 4)
    log(f"  bake: {len(rows):,} texels in {time.time() - started:.1f}s")

    out = trimesh.Trimesh(vertices, faces, process=False)
    out.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            **{k: v for k, v in kwargs.items() if v is not None}))
    return out, record


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------
def solidify(source_path: Path, output: Path, *, faces: int | None = None,
             resolution: int = 512, close: int = 8, texture_size: int = 2048,
             smooth: int = 5, log=print) -> dict:
    """The whole job. Returns a report of what was done and the before/after health."""
    started = time.time()
    source = textured_source(trimesh.load(str(source_path), process=False))
    before = measure(source)
    faces = faces or len(source.faces)
    grid_pitch = float(np.ptp(source.vertices, axis=0).max()) / resolution
    index = SurfaceIndex(source, spacing=grid_pitch * 0.5)
    solid = solidify_geometry(source, faces=faces, resolution=resolution, close=close,
                              smooth=smooth, index=index, log=log)
    after = measure(solid)
    distance, _, _ = index.closest(solid.vertices)
    size = float(np.ptp(source.vertices, axis=0).max())
    result, material = bake_material(solid, source, index, texture_size, log=log)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial.glb")
    result.export(str(temporary), file_type="glb", extension_webp=True,
                  include_normals=True)
    temporary.replace(output)
    report = {
        "source": str(source_path), "output": str(output),
        "settings": {"faces": faces, "resolution": resolution, "close": close,
                     "texture_size": texture_size, "smooth": smooth},
        "before": before, "after": after, "material": material,
        "shift_from_source": {
            "median": round(float(np.median(distance)) / size, 6),
            "p99": round(float(np.percentile(distance, 99)) / size, 6),
            "unit": "fraction of the asset's longest side"},
        "seconds": round(time.time() - started, 1),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="GLB to solidify")
    parser.add_argument("output", type=Path, help="GLB to write")
    parser.add_argument("--faces", type=int, default=None,
                        help="face budget for the result (default: the source's count)")
    parser.add_argument("--resolution", type=int, default=512,
                        help="voxels across the longest side (default 512)")
    parser.add_argument("--close", type=int, default=8,
                        help="seal holes up to about twice this many voxels wide (default 8)")
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--smooth", type=int, default=5,
                        help="Taubin smoothing passes after decimation (default 5)")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source.is_file():
        raise SystemExit(f"missing source: {args.source}")
    if args.close + 4 >= args.resolution:
        raise SystemExit("--close must be well under --resolution")
    report = solidify(args.source, args.output, faces=args.faces,
                      resolution=args.resolution, close=args.close,
                      texture_size=args.texture_size, smooth=args.smooth)
    before, after = report["before"], report["after"]
    print(f"  volume {before['volume']:.4f} -> {after['volume']:.4f}, "
          f"faces {before['faces']:,} -> {after['faces']:,}, "
          f"watertight {before['watertight']} -> {after['watertight']}, "
          f"median shift {report['shift_from_source']['median']:.3%} "
          f"({report['seconds']}s) -> {args.output}")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if after["watertight"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
