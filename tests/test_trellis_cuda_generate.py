"""The NVIDIA TRELLIS.2 generator's torch-free contract: it shares the Mac route's settings,
refuses an unpatched (BRIA-loading) checkout, and writes a complete licence record."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import patch_trellis_cuda_no_bria as bria_patch
import pytest
import trellis_cuda_generate as gen
import trellis_space_generate as space

UPSTREAM_LINE = bria_patch.NEEDLE + "\n"


def _vendor(tmp_path, patched=True, built=True):
    vendor = tmp_path / "trellis-cuda"
    for path in bria_patch.pipeline_files(gen.checkout(vendor)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(UPSTREAM_LINE)
    if patched:
        bria_patch.apply(gen.checkout(vendor))
    (gen.checkout(vendor) / "o-voxel").mkdir()
    (vendor / ".venv" / "bin").mkdir(parents=True)
    (vendor / ".venv" / "bin" / "python").write_text("")
    if built:
        (vendor / gen.BUILT_MARKER_NAME).write_text(json.dumps({"commit": "abc123"}))
    return vendor


def test_same_settings_as_the_mac_route():
    assert gen.DEMO_PARAMS is space.DEMO_PARAMS


def test_backends_are_pinned_not_inherited():
    env = gen.configure_environment({"ATTN_BACKEND": "sdpa", "SPARSE_CONV_BACKEND": "none"})
    assert env["ATTN_BACKEND"] == "flash_attn"
    assert env["SPARSE_ATTN_BACKEND"] == "flash_attn"
    assert env["SPARSE_CONV_BACKEND"] == "flex_gemm"
    assert env["OPENCV_IO_ENABLE_OPENEXR"] == "1"


def test_an_unpatched_checkout_is_refused(tmp_path):
    vendor = _vendor(tmp_path, patched=False)
    refusal = gen.bria_refusal(vendor)
    assert refusal and "BRIA" in refusal and "patch_trellis_cuda_no_bria.py" in refusal


def test_a_patched_checkout_is_allowed(tmp_path):
    assert gen.bria_refusal(_vendor(tmp_path)) is None


def test_generate_refuses_before_touching_the_image(tmp_path, monkeypatch):
    vendor = _vendor(tmp_path, patched=False)
    monkeypatch.setattr(gen, "prepare_image", lambda *a, **k: pytest.fail("matted"))
    monkeypatch.setattr(gen, "load_pipeline", lambda *a: pytest.fail("loaded"))
    with pytest.raises(SystemExit, match="BRIA"):
        gen.generate(tmp_path / "in.png", tmp_path / "out.glb", vendor, seed=0,
                     resolution="1024", decimation_target=1, texture_size=1024,
                     save_latents=False)


def test_load_pipeline_refuses_an_unpatched_checkout_before_importing_torch(tmp_path,
                                                                             monkeypatch):
    vendor = _vendor(tmp_path, patched=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # an import would raise, not pass
    with pytest.raises(SystemExit, match="BRIA"):
        gen.load_pipeline(vendor)


def test_check_reports_the_guardrail(tmp_path, capsys):
    assert gen.check_environment(_vendor(tmp_path, patched=False)) == 1
    assert "BRIA" in capsys.readouterr().out


def test_check_needs_a_finished_build(tmp_path, capsys):
    assert gen.check_environment(_vendor(tmp_path, built=False)) == 1
    assert "completed build" in capsys.readouterr().out


def test_forbid_bria_makes_the_remover_unbuildable():
    rembg = types.ModuleType("trellis2.pipelines.rembg")

    class BiRefNet:
        def __init__(self, model_name="briaai/RMBG-2.0"):
            raise AssertionError("constructor reached: BRIA would download")

    BiRefNet.__module__ = "trellis2.pipelines.rembg.BiRefNet"
    rembg.BiRefNet = BiRefNet
    rembg.Image = type("Image", (), {})  # re-exported from elsewhere: left alone
    gen.forbid_bria(rembg)
    assert rembg.BiRefNet is gen.BriaRefused
    assert rembg.Image is not gen.BriaRefused
    with pytest.raises(RuntimeError, match="BRIA RMBG-2.0 must never be loaded"):
        rembg.BiRefNet(model_name="briaai/RMBG-2.0")


def test_manifest_says_cuda_and_no_rembg():
    manifest = gen.build_manifest(image="in.png", output="out.glb", params=gen.DEMO_PARAMS,
                                  pipeline_type="1024_cascade", seed=0, timings={},
                                  artifacts={}, matte_model="birefnet-general-lite")
    assert manifest["device"] == "cuda"
    assert manifest["generator"] == "trellis_cuda_generate.py"
    assert manifest["load_rembg"] is False
    assert manifest["attn_backend"] == "flash_attn"
    assert manifest["matte_model"] == "birefnet-general-lite"


def test_provenance_records_licences_and_bria_blocked(tmp_path):
    image, output = tmp_path / "in.png", tmp_path / "out.glb"
    image.write_bytes(b"png")
    output.write_bytes(b"glb")
    record = gen.build_provenance(image=image, output=output, parameters={"seed": 0},
                                  matte_model="birefnet-general-lite",
                                  backend_revision="abc123")
    assert record["license"]["name"] == "MIT (TRELLIS.2) + DINOv3 License"
    assert record["model"]["route"] == "nvidia-cuda"
    names = {c["component"]: c for c in record["components"]}
    assert names["BRIA RMBG-2.0"]["loaded"] is False
    assert "rembg/birefnet-general-lite" in names
    assert "facebook/dinov3-vitl16-pretrain-lvd1689m" in names
    assert record["output"]["sha256"] and record["software"]["backend_revision"] == "abc123"
    json.dumps(record)


def test_provenance_without_our_matte_lists_no_remover(tmp_path):
    image, output = tmp_path / "in.png", tmp_path / "out.glb"
    image.write_bytes(b"png")
    output.write_bytes(b"glb")
    record = gen.build_provenance(image=image, output=output, parameters={},
                                  matte_model=None, backend_revision=None)
    assert not any(c["component"].startswith("rembg/") for c in record["components"])


def test_sidecar_paths(tmp_path):
    out = tmp_path / "creature.glb"
    assert gen.provenance_path(out).name == "creature.provenance.json"
    assert gen.matted_path(out).name == "creature__matted.png"


def test_backend_revision_reads_the_build_marker(tmp_path):
    assert gen.backend_revision(_vendor(tmp_path)) == "abc123"
    assert gen.backend_revision(tmp_path / "nothing") is None


def test_an_unmatted_image_is_cut_out_by_our_remover(tmp_path, monkeypatch):
    from PIL import Image

    from image_to_3dlab import matte

    source = tmp_path / "in.png"
    Image.new("RGB", (32, 32), "white").save(source)
    cut = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    monkeypatch.setattr(matte, "cut_out", lambda image: (cut, "birefnet-general-lite"))
    path, model = gen.prepare_image(source, tmp_path / "out.glb", force_matte=False,
                                    allow_uncut=False)
    assert model == "birefnet-general-lite"
    assert path == gen.matted_path(tmp_path / "out.glb") and path.is_file()


def test_a_real_cutout_goes_straight_in(tmp_path, monkeypatch):
    from PIL import Image

    from image_to_3dlab import matte

    source = tmp_path / "in.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (8, 8, 24, 24))
    image.save(source)
    monkeypatch.setattr(matte, "cut_out", lambda *a: pytest.fail("re-matted"))
    assert gen.prepare_image(source, tmp_path / "out.glb", force_matte=False,
                             allow_uncut=False) == (source, None)


def test_main_requires_image_and_output():
    with pytest.raises(SystemExit):
        gen.main([])


# --- AMD: the attention backend the ROCm patches add ------------------------------------

def test_the_nvidia_route_never_asks_for_the_amd_patches(tmp_path):
    assert gen.uses_aule(gen.ROUTE) is False
    assert gen.aule_patch_refusal(_vendor(tmp_path), gen.ROUTE) is None


def test_an_amd_checkout_without_the_attention_patches_is_refused(tmp_path):
    vendor = _vendor(tmp_path)
    refusal = gen.aule_patch_refusal(vendor, gen.trellis_route.ROCM)
    assert refusal and "patch_trellis_rocm.py" in refusal


def test_check_on_amd_reports_the_missing_patches_before_torch(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)  # an import would raise, not pass
    assert gen.check_environment(_vendor(tmp_path), gen.trellis_route.ROCM) == 1
    assert "patch_trellis_rocm.py" in capsys.readouterr().out


def test_check_on_amd_reports_a_missing_aule_before_loading_trellis(tmp_path, monkeypatch,
                                                                   capsys):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "aule", None)
    monkeypatch.setitem(sys.modules, "o_voxel", None)  # reaching TRELLIS would raise
    monkeypatch.setattr(gen.rocm_patch, "is_patched", lambda root: True)
    monkeypatch.setattr(gen, "configure_environment", lambda **k: {})
    assert gen.check_environment(_vendor(tmp_path), gen.trellis_route.ROCM) == 1
    out = capsys.readouterr().out
    assert "aule" in out and "bootstrap_trellis_rocm.py" in out


def test_load_pipeline_on_amd_refuses_unpatched_attention_before_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit, match="patch_trellis_rocm.py"):
        gen.load_pipeline(_vendor(tmp_path), gen.trellis_route.ROCM)


# ---- --solidify --------------------------------------------------------------------------
def test_solidify_is_off_unless_asked():
    assert gen.parse_args(["in.png", "out.glb"]).solidify is False
    assert gen.parse_args(["in.png", "out.glb", "--solidify"]).solidify is True


def test_solidify_command_is_one_the_script_accepts(tmp_path):
    import solidify_mesh

    command = gen.solidify_command(tmp_path / "a.glb", tmp_path / "b.glb", tmp_path / "r.json",
                                   faces=1234, texture_size=1024)
    assert command[:2] == [str(gen.MAIN_PYTHON), str(gen.SOLIDIFY_SCRIPT)]
    parsed = solidify_mesh.parse_args(command[2:])
    assert (parsed.faces, parsed.texture_size) == (1234, 1024)
    assert parsed.report == tmp_path / "r.json"


def test_solidify_environment_names_a_missing_interpreter(tmp_path):
    problem = gen.solidify_environment_problem(tmp_path / "nope" / "python")
    assert "does not exist" in problem


def test_solidify_environment_names_a_missing_import(tmp_path):
    python = tmp_path / "python"
    python.write_text("")
    failed = types.SimpleNamespace(returncode=1,
                                   stderr="Traceback\nModuleNotFoundError: No module named 'xatlas'")
    problem = gen.solidify_environment_problem(python, run=lambda *a, **k: failed)
    assert "xatlas" in problem and "requirements.txt" in problem
    fine = types.SimpleNamespace(returncode=0, stderr="")
    assert gen.solidify_environment_problem(python, run=lambda *a, **k: fine) is None


def _fake_solidify(write: bool):
    def run(command, **_):
        source, output, report = Path(command[2]), Path(command[3]), Path(command[-1])
        assert source.read_bytes() == b"shell"  # it is handed the original
        if write:
            output.write_bytes(b"solid")
            report.write_text(json.dumps({
                "settings": {"faces": 10}, "before": {"volume": 0.005},
                "after": {"volume": 0.29, "watertight": True, "faces": 10},
                "material": {"baked": True}, "shift_from_source": {"median": 0.0007}}))
        return types.SimpleNamespace(returncode=0 if write else 1)
    return run


def test_solidify_output_replaces_the_glb_and_tidies_up(tmp_path):
    out = tmp_path / "rock.glb"
    out.write_bytes(b"shell")
    record = gen.solidify_output(out, faces=10, texture_size=256, run=_fake_solidify(True))
    assert out.read_bytes() == b"solid"
    assert record["applied"] and record["watertight"]
    assert record["volume"] == {"before": 0.005, "after": 0.29}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rock.glb"]
    json.dumps(record)


def test_a_failed_solidify_keeps_the_original(tmp_path, capsys):
    out = tmp_path / "rock.glb"
    out.write_bytes(b"shell")
    record = gen.solidify_output(out, faces=10, texture_size=256, run=_fake_solidify(False))
    assert out.read_bytes() == b"shell"
    assert record["applied"] is False and "without output" in record["error"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rock.glb"]
    assert "original kept" in capsys.readouterr().out


def test_maybe_solidify_records_into_params_and_timings(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "solidify_output",
                        lambda output, **_: {"applied": True, "seconds": 3.0})
    timings = {}
    assert gen._maybe_solidify(tmp_path / "x.glb", False, decimation_target=1,
                               texture_size=1, timings=timings) == {}
    assert timings == {}
    extra = gen._maybe_solidify(tmp_path / "x.glb", True, decimation_target=1,
                                texture_size=1, timings=timings)
    assert extra == {"solidify": {"applied": True, "seconds": 3.0}}
    assert timings == {"solidify": 3.0}


def test_solidify_refuses_before_the_model_loads(tmp_path, monkeypatch):
    image = tmp_path / "in.png"
    image.write_bytes(b"")
    monkeypatch.setattr(gen, "solidify_environment_problem", lambda: "no xatlas here")
    monkeypatch.setattr(gen, "generate", lambda *a, **k: pytest.fail("generated anyway"))
    with pytest.raises(SystemExit, match="no xatlas here"):
        gen.main([str(image), str(tmp_path / "out.glb"), "--solidify"])
