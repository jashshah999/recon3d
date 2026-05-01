"""End-to-end test using VGGT for pose estimation + gsplat training.

This is the real integration test: images -> VGGT -> gsplat -> .ply
Requires GPU, VGGT model weights, and gsplat.
"""

import numpy as np
import cv2
import torch
import pytest
from pathlib import Path


def _make_textured_scene(output_dir: Path, n_images: int = 6):
    """Create a synthetic multi-view dataset with more texture for VGGT."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H, W = 512, 512
    fx = fy = 400.0
    cx, cy = W / 2, H / 2

    np.random.seed(42)
    n_pts = 2000
    points = np.random.randn(n_pts, 3).astype(np.float32) * 0.5
    colors = np.random.rand(n_pts, 3).astype(np.float32)

    image_paths = []
    extrinsics_gt = []

    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 3.0
        cam_pos = np.array([
            radius * np.cos(angle),
            0.5 * np.sin(angle * 0.5),
            radius * np.sin(angle),
        ])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        extrinsics_gt.append(w2c)

        pts_cam = (R @ points.T).T + t
        z = pts_cam[:, 2]
        valid = z > 0.1

        u = (fx * pts_cam[:, 0] / z + cx).astype(int)
        v = (fy * pts_cam[:, 1] / z + cy).astype(int)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Add checkerboard background for texture
        for row in range(0, H, 32):
            for col in range(0, W, 32):
                if (row // 32 + col // 32) % 2 == 0:
                    img[row:row+32, col:col+32] = [40, 40, 40]
                else:
                    img[row:row+32, col:col+32] = [60, 60, 60]

        for j in range(len(points)):
            if valid[j] and 0 <= u[j] < W and 0 <= v[j] < H:
                c = (colors[j] * 255).astype(np.uint8)
                cv2.circle(img, (u[j], v[j]), 4, c.tolist(), -1)

        path = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img)
        image_paths.append(str(path))

    return image_paths, np.array(extrinsics_gt)


@pytest.fixture(autouse=True)
def cleanup_gpu():
    yield
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_vggt_pose_estimation(tmp_path):
    """Test VGGT pose estimation on synthetic images."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        pytest.skip("VGGT not installed")

    from recon3d.pose_estimation import estimate_poses_vggt

    image_paths, gt_extrinsics = _make_textured_scene(tmp_path, n_images=6)

    result = estimate_poses_vggt(
        image_paths, device="cuda", conf_threshold=0.5
    )

    assert result.extrinsics.shape == (6, 4, 4)
    assert result.intrinsics.shape == (6, 3, 3)
    assert result.point_cloud.shape[1] == 3
    assert result.point_colors.shape[1] == 3
    assert len(result.point_cloud) > 0
    assert not result.is_metric

    print(f"Got {len(result.point_cloud)} points from VGGT")
    print(f"Extrinsics shape: {result.extrinsics.shape}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_vggt_to_gsplat_pipeline(tmp_path):
    """Full pipeline: VGGT poses -> gsplat training -> PLY export."""
    try:
        from vggt.models.vggt import VGGT
        from gsplat.rendering import rasterization
    except ImportError:
        pytest.skip("VGGT and gsplat required")

    from recon3d.pose_estimation import estimate_poses_vggt
    from recon3d.gaussian_train import train_gaussians, TrainConfig

    image_paths, _ = _make_textured_scene(tmp_path, n_images=6)

    print("Running VGGT...")
    result = estimate_poses_vggt(
        image_paths, device="cuda", conf_threshold=0.5
    )

    print(f"VGGT returned {len(result.point_cloud)} points")

    config = TrainConfig(max_steps=300, log_every=100, densify=False)

    print("Training gsplat...")
    ply_path = train_gaussians(
        image_paths=image_paths,
        extrinsics=result.extrinsics,
        intrinsics=result.intrinsics,
        point_cloud=result.point_cloud,
        point_colors=result.point_colors,
        output_dir=str(tmp_path / "output"),
        config=config,
        device="cuda",
    )

    assert Path(ply_path).exists()
    assert (tmp_path / "output" / "scene.splat").exists()
    print(f"Pipeline complete! Output: {ply_path}")
