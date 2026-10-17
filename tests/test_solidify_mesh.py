"""`solidify_mesh.py`: a hollow shell with a hole comes out solid and sealed, and the
colour that comes with it is read from where the original actually had it.

Every helper the script uses is tested here on small synthetic meshes, so a defect costs
a second rather than a TRELLIS run. The fixture is the defect itself in miniature: a
sphere made of two skins a few voxels apart, with a hole punched through both.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import solidify_mesh as sm
import trimesh
from PIL import Image


def _holed_shell(radius: float = 1.0, thickness: float = 0.04,
                 hole_angle: float = 0.1) -> trimesh.Trimesh:
    """Outer skin facing out, inner skin facing in, a hole through both at +z."""
    outer = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    inner = trimesh.creation.icosphere(subdivisions=4, radius=radius - thickness)
    inner.invert()
    parts = []
    for skin in (outer, inner):
        direction = skin.triangles_center / np.linalg.norm(skin.triangles_center, axis=1,
                                                           keepdims=True)
        keep = direction[:, 2] < np.cos(hole_angle)
        parts.append(trimesh.Trimesh(skin.vertices, skin.faces[keep], process=False))
    shell = trimesh.util.concatenate(parts)
    assert not shell.is_watertight  # the hole really is there
    return shell


def _sphere_volume(radius: float = 1.0) -> float:
    return 4.0 / 3.0 * np.pi * radius ** 3


# ---- geometry helpers -------------------------------------------------------------------
def test_closest_on_triangles_matches_trimesh():
    rng = np.random.default_rng(1)
    triangles = rng.normal(size=(500, 3, 3))
    points = rng.normal(size=(500, 3)) * 2  # plenty land in vertex and edge regions
    ours, bary = sm.closest_on_triangles(points, *triangles.transpose(1, 0, 2))
    reference = trimesh.triangles.closest_point(triangles, points)
    np.testing.assert_allclose(ours, reference, atol=1e-9)
    np.testing.assert_allclose(bary.sum(axis=1), 1.0)
    assert (bary >= -1e-12).all()


def test_closest_on_a_degenerate_triangle_is_finite():
    a = np.zeros((1, 3))
    point, bary = sm.closest_on_triangles(np.ones((1, 3)), a, a, a)
    assert np.isfinite(point).all() and np.isfinite(bary).all()


def test_surface_index_distance_matches_trimesh():
    """Exact near the surface, which is where the script asks: the signed field only
    queries a few voxels either side of it, and the bake only queries points on it.
    Random samples leave gaps, so a few queries miss the true nearest face; the class
    docstring gives the measured bound and this pins it."""
    mesh = trimesh.creation.icosphere(subdivisions=3)
    spacing = 0.02
    index = sm.SurfaceIndex(mesh, spacing=spacing)
    rng = np.random.default_rng(2)
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    near = direction * rng.uniform(1 - 3 * spacing, 1 + 3 * spacing, size=(2000, 1))
    distance, face, bary = index.closest(near)
    _, reference, _ = trimesh.proximity.closest_point(mesh, near)
    error = distance - reference
    assert (error > -1e-9).all()  # never closer than the truth: every answer is real
    assert (error < 1e-9).mean() > 0.99
    assert error.max() < spacing / 4
    far = rng.normal(size=(300, 3)) * 2
    far_distance, _, _ = index.closest(far)
    np.testing.assert_allclose(far_distance,
                               trimesh.proximity.closest_point(mesh, far)[1], atol=spacing)
    queries = near
    rebuilt = np.einsum("nij,ni->nj", mesh.triangles[face], bary)
    np.testing.assert_allclose(np.linalg.norm(rebuilt - queries, axis=1), distance,
                               atol=1e-9)


def test_grid_is_padded_and_cubic():
    grid = sm.grid_for(np.array([[0.0, 0, 0], [2, 1, 1]]), resolution=20, pad=3)
    assert grid.pitch == pytest.approx(0.1)
    assert grid.shape == (27, 17, 17)
    np.testing.assert_allclose(grid.to_world(grid.to_index(np.array([[0.5, 0.5, 0.5]]))),
                               [[0.5, 0.5, 0.5]])


def test_without_closing_the_flood_pours_through_the_hole():
    shell = _holed_shell()
    grid = sm.grid_for(shell.bounds, 64, pad=8)
    surface = sm.surface_mask(sm.SurfaceIndex(shell, grid.pitch * 0.5).points, grid)
    open_fill = sm.solid_mask(surface, close=0).sum() * grid.pitch ** 3
    closed_fill = sm.solid_mask(surface, close=4).sum() * grid.pitch ** 3
    assert open_fill < 0.3 * _sphere_volume()  # only the wall: the cavity leaked
    assert closed_fill == pytest.approx(_sphere_volume(), rel=0.1)


def test_solid_mask_refuses_a_grid_with_no_outside():
    with pytest.raises(ValueError):
        sm.solid_mask(np.ones((5, 5, 5), dtype=bool), close=0)


def test_a_holed_hollow_shell_comes_out_solid_and_watertight():
    shell = _holed_shell()
    solid = sm.solidify_geometry(shell, faces=4000, resolution=64, close=4,
                                 log=lambda *_: None)
    assert solid.is_watertight
    assert solid.body_count == 1
    assert solid.volume == pytest.approx(_sphere_volume(), rel=0.05)
    assert len(solid.faces) <= 4000
    # The skin sits on the original outer surface, not a voxel away from it.
    radius = np.linalg.norm(solid.vertices, axis=1)
    assert np.median(np.abs(radius - 1.0)) < 64 ** -1 * 2 * 0.5


# ---- texture helpers --------------------------------------------------------------------
def test_rasterize_covers_every_texel_once():
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    rows, cols, face, bary = sm.rasterize_uv(uv, faces, 16)
    assert len(rows) == 256
    assert len(np.unique(rows * 16 + cols)) == 256
    # Barycentrics rebuild the texel centre each was found at.
    rebuilt = sm.uv_to_pixel(np.einsum("nij,ni->nj", uv[faces[face]], bary), 16)
    np.testing.assert_allclose(rebuilt, np.stack([cols, rows], axis=1), atol=1e-6)


def test_rasterize_skips_zero_area_triangles():
    uv = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    rows, _, _, _ = sm.rasterize_uv(uv, np.array([[0, 1, 2]]), 32)
    assert len(rows) == 0


def test_sampler_reads_exactly_where_the_rasteriser_writes():
    image = np.random.default_rng(3).integers(0, 255, size=(8, 8, 3)).astype(np.float32)
    rows, cols = np.mgrid[0:8, 0:8]
    rows, cols = rows.ravel(), cols.ravel()
    uv = np.stack([(cols + 0.5) / 8, 1 - (rows + 0.5) / 8], axis=1)
    np.testing.assert_allclose(sm.sample_bilinear(image, uv), image[rows, cols])


def test_v_is_up_like_trimesh():
    """uv (0, 1) is the top-left of the picture, as trimesh's own `uv_to_color` reads it."""
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[0, 0] = 255
    assert sm.sample_bilinear(image, np.array([[0.0, 1.0]]))[0, 0] == 255
    assert sm.sample_bilinear(image, np.array([[0.0, 0.0]]))[0, 0] == 0


def test_gutters_copy_their_nearest_texel():
    image = np.zeros((1, 5, 1))
    image[0, 1] = 7
    filled = np.zeros((1, 5), dtype=bool)
    filled[0, 1] = True
    assert sm.fill_gutters(image, filled).ravel().tolist() == [7] * 5


def test_flat_metallic_roughness_becomes_numbers():
    flat = np.zeros((16, 16, 3), dtype=np.uint8)
    flat[..., 1], flat[..., 2] = 220, 0
    assert sm.flat_value(flat, (1, 2)) == pytest.approx((220 / 255, 0.0))
    varied = flat.copy()
    varied[:8, :, 1] = 40
    assert sm.flat_value(varied, (1, 2)) is None


# ---- end to end -------------------------------------------------------------------------
def _gradient(size: int) -> np.ndarray:
    """Red follows u, green follows v (v up): any flip or offset shows up as wrong colour."""
    x = (np.arange(size) + 0.5) / size
    red = np.tile(x * 255, (size, 1))
    green = np.tile((1 - x)[:, None] * 255, (1, size))
    return np.stack([red, green, np.full_like(red, 128)], axis=-1).astype(np.uint8)


def _textured_shell_glb(path, size=256, flat_mr=True):
    shell = _holed_shell()
    vmap, faces, uv = sm.unwrap(shell, size)
    mr = np.zeros((size, size, 3), dtype=np.uint8)
    mr[..., 1] = 200
    if not flat_mr:
        mr[: size // 2, :, 1] = 30
    textured = trimesh.Trimesh(shell.vertices[vmap], faces, process=False)
    textured.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(_gradient(size)),
            metallicRoughnessTexture=Image.fromarray(mr),
            metallicFactor=1.0, roughnessFactor=1.0))
    textured.export(str(path))
    return textured


def test_solidify_end_to_end(tmp_path):
    source = _textured_shell_glb(tmp_path / "shell.glb")
    out = tmp_path / "solid.glb"
    report = sm.solidify(tmp_path / "shell.glb", out, faces=6000, resolution=64, close=4,
                         texture_size=256, log=lambda *_: None)
    assert report["after"]["watertight"] and not report["before"]["watertight"]
    assert report["after"]["volume"] == pytest.approx(_sphere_volume(), rel=0.05)
    assert report["material"]["metallic_roughness"] == "flat, replaced by factors"
    json.dumps(report)  # the route writes it into the manifest

    result = trimesh.load(str(out), force="mesh", process=False)
    material = result.visual.material
    assert material.metallicRoughnessTexture is None
    assert material.roughnessFactor == pytest.approx(200 / 255, abs=0.01)
    assert material.metallicFactor == pytest.approx(0.0, abs=0.01)

    # Colour read back by trimesh's own reader matches the source's at the same spot.
    # Points away from the hole only: the patch has no original colour of its own.
    baked = result.visual.to_color().vertex_colors[:, :3].astype(float)
    keep = result.vertices[:, 2] < 0.9
    expected = source.visual.to_color().vertex_colors[:, :3].astype(float)
    _, nearest = trimesh.proximity.ProximityQuery(source).vertex(result.vertices[keep])
    error = np.abs(baked[keep] - expected[nearest]).max(axis=1)
    assert np.median(error) < 12


def test_a_varied_metallic_roughness_map_is_rebaked(tmp_path):
    _textured_shell_glb(tmp_path / "shell.glb", flat_mr=False)
    report = sm.solidify(tmp_path / "shell.glb", tmp_path / "solid.glb", faces=4000,
                         resolution=48, close=4, texture_size=128, log=lambda *_: None)
    assert report["material"]["metallic_roughness"] == "re-baked"
    result = trimesh.load(str(tmp_path / "solid.glb"), force="mesh", process=False)
    assert result.visual.material.metallicRoughnessTexture is not None


def test_an_untextured_source_still_solidifies(tmp_path):
    _holed_shell().export(str(tmp_path / "shell.glb"))
    report = sm.solidify(tmp_path / "shell.glb", tmp_path / "solid.glb", faces=3000,
                         resolution=48, close=4, log=lambda *_: None)
    assert report["material"] == {"baked": False}
    assert report["after"]["watertight"]


def test_main_writes_a_report_and_exits_zero_when_sealed(tmp_path):
    _holed_shell().export(str(tmp_path / "shell.glb"))
    code = sm.main([str(tmp_path / "shell.glb"), str(tmp_path / "solid.glb"),
                    "--resolution", "48", "--close", "4", "--faces", "3000",
                    "--report", str(tmp_path / "report.json")])
    assert code == 0
    assert json.loads((tmp_path / "report.json").read_text())["after"]["watertight"]
    assert not (tmp_path / "solid.glb.partial.glb").exists()


def test_main_refuses_a_missing_source(tmp_path):
    with pytest.raises(SystemExit):
        sm.main([str(tmp_path / "nope.glb"), str(tmp_path / "out.glb")])
