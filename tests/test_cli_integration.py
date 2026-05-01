"""Integration test: run the full CLI pipeline on synthetic multi-view images."""

import numpy as np
import cv2
import torch
import pytest
from pathlib import Path


def _create_multiview_images(output_dir: Path, n_images: int = 8):
    """Create synthetic multi-view images with real 3D parallax."""
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = 512, 512
    fx = fy = 400.0
    cx, cy = W / 2, H / 2

    np.random.seed(42)
    n_pts = 3000
    points = np.random.randn(n_pts, 3).astype(np.float32) * 0.8
    colors = np.random.rand(n_pts, 3).astype(np.float32)
    # Add some structured geometry
    for k in range(500):
        angle = 2 * np.pi * k / 500
        r = 0.3
        points = np.vstack([points, [r * np.cos(angle), 0.5, r * np.sin(angle)]])
        colors = np.vstack([colors, [1.0, 0.2, 0.2]])
    for k in range(500):
        angle = 2 * np.pi * k / 500
        r = 0.4
        points = np.vstack([points, [r * np.cos(angle), -0.3, r * np.sin(angle)]])
        colors = np.vstack([colors, [0.2, 0.2, 1.0]])

    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 3.0
        cam_pos = np.array([
            radius * np.cos(angle),
            0.8 * np.sin(angle * 0.7),
            radius * np.sin(angle),
        ])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        pts_cam = (R @ points.T).T + t
        z = pts_cam[:, 2]
        valid = z > 0.1

        u = (fx * pts_cam[:, 0] / z + cx).astype(int)
        v = (fy * pts_cam[:, 1] / z + cy).astype(int)

        # textured background
        img = np.zeros((H, W, 3), dtype=np.uint8)
        for r in range(0, H, 16):
            for c in range(0, W, 16):
                base = 30 + ((r * 7 + c * 13) % 40)
                img[r:r+16, c:c+16] = [base, base + 10, base + 5]

        for j in range(len(points)):
            if valid[j] and 0 <= u[j] < W and 0 <= v[j] < H:
                c = (np.clip(colors[j], 0, 1) * 255).astype(np.uint8)
                cv2.circle(img, (u[j], v[j]), 5, c.tolist(), -1)

        path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return output_dir


@pytest.fixture(autouse=True)
def cleanup_gpu():
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_cli_full_pipeline(tmp_path):
    """Test the full CLI pipeline: images dir -> VGGT -> gsplat -> outputs."""
    try:
        from vggt.models.vggt import VGGT
        from gsplat.rendering import rasterization
    except ImportError:
        pytest.skip("VGGT and gsplat required")

    from click.testing import CliRunner
    from recon3d.cli import main

    img_dir = _create_multiview_images(tmp_path / "images", n_images=8)
    output_dir = tmp_path / "output"

    runner = CliRunner()
    result = runner.invoke(main, [
        "run", str(img_dir),
        "-o", str(output_dir),
        "--steps", "300",
        "--no-viewer",
        "--no-metric",
    ])

    print(result.output)
    if result.exception:
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

    assert result.exit_code == 0, f"CLI failed: {result.output}"
    assert (output_dir / "scene.ply").exists()
    assert (output_dir / "scene.splat").exists()
    assert (output_dir / "checkpoint.pt").exists()

    ply_size = (output_dir / "scene.ply").stat().st_size
    assert ply_size > 1000, f"PLY file too small: {ply_size} bytes"
    print(f"PLY size: {ply_size / 1024:.1f} KB")
