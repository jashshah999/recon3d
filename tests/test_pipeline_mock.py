"""Mock tests for the pipeline that don't require GPU or model weights."""

import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

from recon3d.pose_estimation import PoseEstimationResult
from recon3d.gaussian_train import _rgb_to_sh, TrainConfig


@pytest.fixture
def mock_pose_result():
    """Create a mock PoseEstimationResult."""
    n_images = 5
    n_points = 1000

    extrinsics = np.zeros((n_images, 4, 4))
    for i in range(n_images):
        extrinsics[i] = np.eye(4)
        extrinsics[i, 2, 3] = i * 0.5

    intrinsics = np.zeros((n_images, 3, 3))
    for i in range(n_images):
        intrinsics[i] = np.array([
            [500, 0, 160],
            [0, 500, 120],
            [0, 0, 1],
        ])

    points = np.random.randn(n_points, 3).astype(np.float32)
    colors = np.random.rand(n_points, 3).astype(np.float32)
    depths = [np.random.rand(240, 320).astype(np.float32) * 5 for _ in range(n_images)]

    return PoseEstimationResult(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=points,
        point_colors=colors,
        depth_maps=depths,
        image_sizes=[(240, 320)] * n_images,
        is_metric=False,
    )


def test_pose_estimation_result_shapes(mock_pose_result):
    r = mock_pose_result
    assert r.extrinsics.shape == (5, 4, 4)
    assert r.intrinsics.shape == (5, 3, 3)
    assert r.point_cloud.shape[1] == 3
    assert r.point_colors.shape[1] == 3
    assert len(r.depth_maps) == 5
    assert not r.is_metric


def test_rgb_to_sh():
    rgb = np.array([[1.0, 0.0, 0.5]])
    import torch
    sh = _rgb_to_sh(torch.from_numpy(rgb).float())
    C0 = 0.28209479177387814
    expected = (rgb - 0.5) / C0
    np.testing.assert_allclose(sh.numpy(), expected, atol=1e-6)


def test_train_config_defaults():
    config = TrainConfig()
    assert config.max_steps == 7000
    assert config.sh_degree == 3
    assert config.ssim_weight == 0.2


def test_pipeline_config():
    from recon3d.pipeline import PipelineConfig
    config = PipelineConfig()
    assert config.pose_method == "vggt"
    assert config.max_frames == 80
    assert config.metric_align is True
    assert config.resize_long_edge == 960


def test_cli_exists():
    from recon3d.cli import main
    assert main is not None
