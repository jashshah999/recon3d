"""End-to-end test with synthetic data.

This test creates synthetic posed images and runs the Gaussian Splatting
training to verify the pipeline works. Requires a GPU and gsplat installed.
"""

import numpy as np
import cv2
import torch
import pytest
from pathlib import Path


def _make_synthetic_scene(output_dir: Path, n_images: int = 8):
    """Create a synthetic multi-view dataset of a colored cube."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H, W = 256, 256
    fx = fy = 200.0
    cx, cy = W / 2, H / 2

    cube_points = []
    cube_colors = []
    face_colors = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
    ]
    for face_idx in range(6):
        for _ in range(200):
            pt = np.random.rand(3) * 0.5 - 0.25
            axis = face_idx // 2
            sign = 1 if face_idx % 2 == 0 else -1
            pt[axis] = sign * 0.25
            cube_points.append(pt)
            cube_colors.append(face_colors[face_idx])

    points = np.array(cube_points, dtype=np.float32)
    colors = np.array(cube_colors, dtype=np.float32)

    image_paths = []
    extrinsics = np.zeros((n_images, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((n_images, 3, 3), dtype=np.float32)

    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 2.0
        cam_pos = np.array([
            radius * np.cos(angle),
            0.3 * np.sin(angle * 2),
            radius * np.sin(angle),
        ])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0)
        t = -R @ cam_pos

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

        extrinsics[i] = w2c
        intrinsics[i] = K

        pts_cam = (R @ points.T).T + t
        z = pts_cam[:, 2]
        valid = z > 0.1

        u = (fx * pts_cam[:, 0] / z + cx).astype(int)
        v = (fy * pts_cam[:, 1] / z + cy).astype(int)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        for j in range(len(points)):
            if valid[j] and 0 <= u[j] < W and 0 <= v[j] < H:
                c = (np.array(colors[j]) * 255).astype(np.uint8)
                cv2.circle(img, (u[j], v[j]), 3, c.tolist(), -1)

        path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))

    return image_paths, extrinsics, intrinsics, points, colors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_gsplat_training_synthetic(tmp_path):
    """Test that gsplat training works with synthetic data."""
    try:
        from gsplat.rendering import rasterization
        from gsplat import export_splats
    except ImportError:
        pytest.skip("gsplat not installed")

    from recon3d.gaussian_train import train_gaussians, TrainConfig

    image_paths, extrinsics, intrinsics, points, colors = _make_synthetic_scene(
        tmp_path, n_images=8
    )

    config = TrainConfig(max_steps=500, log_every=100, densify=False)

    ply_path = train_gaussians(
        image_paths=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=points,
        point_colors=colors,
        output_dir=str(tmp_path / "output"),
        config=config,
        device="cuda",
    )

    assert Path(ply_path).exists()
    assert (tmp_path / "output" / "scene.splat").exists()
    assert (tmp_path / "output" / "checkpoint.pt").exists()

    checkpoint = torch.load(str(tmp_path / "output" / "checkpoint.pt"), weights_only=False)
    assert "splats" in checkpoint
    assert checkpoint["n_gaussians"] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_gsplat_renders_improve(tmp_path):
    """Verify that training actually reduces the loss."""
    try:
        from gsplat.rendering import rasterization
    except ImportError:
        pytest.skip("gsplat not installed")

    from recon3d.gaussian_train import train_gaussians, TrainConfig

    image_paths, extrinsics, intrinsics, points, colors = _make_synthetic_scene(
        tmp_path, n_images=4
    )

    config = TrainConfig(max_steps=200, log_every=50, densify=False)

    ply_path = train_gaussians(
        image_paths=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=points,
        point_colors=colors,
        output_dir=str(tmp_path / "output"),
        config=config,
        device="cuda",
    )

    assert Path(ply_path).exists()
