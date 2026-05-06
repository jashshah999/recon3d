"""Tests for chunked VGGT pipeline components."""

import numpy as np
import pytest


def test_procrustes_sim3_identity():
    from recon3d.chunked_vggt import _procrustes_sim3
    pts = np.random.randn(10, 3)
    T, scale = _procrustes_sim3(pts, pts)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(scale, 1.0, atol=1e-6)
    np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-6)


def test_procrustes_sim3_known_transform():
    from recon3d.chunked_vggt import _procrustes_sim3
    np.random.seed(42)
    src = np.random.randn(20, 3)
    R_true = np.eye(3)
    scale_true = 2.5
    t_true = np.array([1.0, -0.5, 0.3])
    dst = scale_true * (R_true @ src.T).T + t_true

    T, scale = _procrustes_sim3(src, dst)
    np.testing.assert_allclose(scale, scale_true, atol=1e-4)
    np.testing.assert_allclose(T[:3, 3], t_true, atol=1e-4)


def test_naive_stitch_single_chunk():
    from recon3d.chunked_vggt import _naive_stitch

    N = 5
    c2w = np.zeros((N, 4, 4))
    for i in range(N):
        c2w[i] = np.eye(4)
        c2w[i, 2, 3] = i * 0.1

    chunks = [{
        "start": 0, "end": N, "c2w": c2w,
        "pose_conf": np.ones(N),
    }]

    result = _naive_stitch(chunks, N)
    np.testing.assert_allclose(result, c2w, atol=1e-6)


def test_naive_stitch_two_chunks():
    from recon3d.chunked_vggt import _naive_stitch

    N = 8
    overlap = 3

    chunk1_c2w = np.zeros((5, 4, 4))
    for i in range(5):
        chunk1_c2w[i] = np.eye(4)
        chunk1_c2w[i, 2, 3] = i * 0.1

    chunk2_c2w = np.zeros((6, 4, 4))
    for i in range(6):
        chunk2_c2w[i] = np.eye(4)
        chunk2_c2w[i, 2, 3] = (i + 2) * 0.1

    chunks = [
        {"start": 0, "end": 5, "c2w": chunk1_c2w, "pose_conf": np.ones(5)},
        {"start": 2, "end": 8, "c2w": chunk2_c2w, "pose_conf": np.ones(6)},
    ]

    result = _naive_stitch(chunks, N)
    assert result.shape == (N, 4, 4)
    # All poses should be valid (not identity)
    for i in range(N):
        assert not np.allclose(result[i], np.eye(4)) or i == 0


def test_interpolate_poses():
    from recon3d.chunked_vggt import _interpolate_poses

    p1 = np.eye(4)
    p1[:3, 3] = [0, 0, 0]
    p2 = np.eye(4)
    p2[:3, 3] = [1, 0, 0]

    result = _interpolate_poses(p1, p2, 0.5)
    np.testing.assert_allclose(result[:3, 3], [0.5, 0, 0], atol=1e-6)
    np.testing.assert_allclose(result[:3, :3], np.eye(3), atol=1e-6)


def test_pipeline_config_new_fields():
    from recon3d.pipeline import PipelineConfig
    config = PipelineConfig()
    assert config.chunk_size == 20
    assert config.chunk_overlap == 5
    assert config.use_factor_graph is True
    assert config.export_mesh is False
