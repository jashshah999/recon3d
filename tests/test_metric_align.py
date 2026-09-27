"""Tests for MoGe model selection in metric alignment (no model download needed)."""

import sys
import types
from typing import ClassVar

import cv2
import numpy as np
import pytest
import torch
from click.testing import CliRunner

from recon3d.metric_align import align_to_metric, resolve_metric_model


class FakeMoGe:
    loaded: ClassVar[list] = []
    calls: ClassVar[list] = []

    @classmethod
    def from_pretrained(cls, name):
        cls.loaded.append(name)
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def infer(self, image, **kwargs):
        FakeMoGe.calls.append(kwargs)
        h, w = image.shape[1:]
        # Metric depth is exactly 2x the relative depth used in the tests.
        return {"depth": torch.full((h, w), 4.0, dtype=torch.float16)}


@pytest.fixture
def fake_moge(monkeypatch):
    FakeMoGe.loaded, FakeMoGe.calls = [], []
    pkg = types.ModuleType("moge")
    model_pkg = types.ModuleType("moge.model")
    monkeypatch.setitem(sys.modules, "moge", pkg)
    monkeypatch.setitem(sys.modules, "moge.model", model_pkg)
    for version in ("v2", "v3"):
        mod = types.ModuleType(f"moge.model.{version}")
        mod.MoGeModel = type(f"MoGe{version}", (FakeMoGe,), {})
        monkeypatch.setitem(sys.modules, f"moge.model.{version}", mod)
        setattr(model_pkg, version, mod)
    return FakeMoGe


def _inputs(tmp_path, n=3):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), np.full((24, 32, 3), 100, dtype=np.uint8))
        paths.append(str(p))
    extr = np.tile(np.eye(4), (n, 1, 1))
    extr[:, :3, 3] = [1.0, 2.0, 3.0]
    intr = np.tile(np.eye(3), (n, 1, 1))
    points = np.ones((10, 3))
    depths = [np.full((24, 32), 2.0) for _ in range(n)]
    return paths, extr, intr, points, depths


def test_resolve_metric_model():
    assert resolve_metric_model("moge-2") == ("v2", "Ruicheng/moge-2-vitl")
    assert resolve_metric_model("moge-3") == ("v3", "Ruicheng/moge-3-vitl")
    assert resolve_metric_model("moge-3-vitg") == ("v3", "Ruicheng/moge-3-vitg")
    assert resolve_metric_model("Ruicheng/moge-2-vitb-normal") == ("v2", "Ruicheng/moge-2-vitb-normal")
    assert resolve_metric_model("/ckpts/moge-3-vitl.pt") == ("v3", "/ckpts/moge-3-vitl.pt")
    with pytest.raises(ValueError, match="MoGe-1 has no metric scale"):
        resolve_metric_model("Ruicheng/moge-vitl")


@pytest.mark.parametrize(
    "name, device, repo, kwargs",
    [
        ("moge-2", "cuda", "Ruicheng/moge-2-vitl", {"use_fp16": True}),
        ("moge-2", "cpu", "Ruicheng/moge-2-vitl", {"use_fp16": False}),
        ("moge-3", "cuda", "Ruicheng/moge-3-vitl", {}),
        ("moge-3", "cpu", "Ruicheng/moge-3-vitl", {"refine_steps": 0}),
    ],
)
def test_align_uses_selected_model(tmp_path, fake_moge, monkeypatch, name, device, repo, kwargs):
    # Only the choice of infer() flags depends on the device; keep tensors on CPU.
    real_tensor = torch.tensor
    monkeypatch.setattr("recon3d.metric_align.torch.tensor",
                        lambda *a, device=None, **k: real_tensor(*a, **k))
    paths, extr, intr, points, depths = _inputs(tmp_path)
    out_extr, _, out_points, out_depths = align_to_metric(
        paths, extr, intr, points, depths, device=device, metric_model=name
    )
    assert fake_moge.loaded == [repo]
    assert fake_moge.calls == [kwargs] * 3
    np.testing.assert_allclose(out_extr[:, :3, 3], extr[:, :3, 3] * 2.0)
    np.testing.assert_allclose(out_points, points * 2.0)
    np.testing.assert_allclose(out_depths[0], depths[0] * 2.0)


def test_missing_moge3_module_explains(tmp_path, fake_moge, monkeypatch):
    monkeypatch.setitem(sys.modules, "moge.model.v3", None)
    paths, extr, intr, points, depths = _inputs(tmp_path)
    with pytest.raises(ImportError, match="MoGe >= 3.0"):
        align_to_metric(paths, extr, intr, points, depths, device="cpu", metric_model="moge-3")


def test_moge3_without_gpu_explains(tmp_path, fake_moge, monkeypatch):
    real_import = __import__("importlib").import_module

    def fake_import(name):
        if name == "moge.model.v3":
            raise RuntimeError("No CUDA GPUs are available")
        return real_import(name)

    monkeypatch.setattr("recon3d.metric_align.importlib.import_module", fake_import)
    paths, extr, intr, points, depths = _inputs(tmp_path)
    with pytest.raises(ImportError, match="No CUDA GPUs are available.*CUDA GPU"):
        align_to_metric(paths, extr, intr, points, depths, device="cpu", metric_model="moge-3")


def test_cli_rejects_unknown_metric_model(tmp_path):
    from recon3d.cli import main

    (tmp_path / "imgs").mkdir()
    result = CliRunner().invoke(main, ["run", str(tmp_path / "imgs"), "--metric-model", "depth-anything"])
    assert result.exit_code == 2
    assert "--metric-model" in result.output
