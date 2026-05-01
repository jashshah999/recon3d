"""Tests for video frame extraction."""

import cv2
import numpy as np
import pytest
import tempfile
from pathlib import Path

from recon3d.video import extract_frames, extract_frames_from_directory


@pytest.fixture
def sample_video(tmp_path):
    """Create a simple test video."""
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))

    for i in range(90):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    return str(video_path)


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a directory of test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(20):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.putText(img, f"Image {i}", (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)

    return str(img_dir)


def test_extract_frames_basic(sample_video, tmp_path):
    output_dir = tmp_path / "frames"
    paths = extract_frames(sample_video, str(output_dir), max_frames=10, min_blur_score=0.0)

    assert len(paths) > 0
    assert len(paths) <= 10
    for p in paths:
        assert Path(p).exists()
        img = cv2.imread(p)
        assert img is not None


def test_extract_frames_blur_filter(sample_video, tmp_path):
    output_dir = tmp_path / "frames"
    paths_low = extract_frames(
        sample_video, str(output_dir), max_frames=50, min_blur_score=0.0
    )
    output_dir2 = tmp_path / "frames2"
    paths_high = extract_frames(
        sample_video, str(output_dir2), max_frames=50, min_blur_score=10000.0
    )

    assert len(paths_low) >= len(paths_high)


def test_extract_frames_resize(sample_video, tmp_path):
    output_dir = tmp_path / "frames"
    paths = extract_frames(
        sample_video, str(output_dir), max_frames=5,
        min_blur_score=0.0, resize_long_edge=160
    )

    assert len(paths) > 0
    img = cv2.imread(paths[0])
    h, w = img.shape[:2]
    assert max(h, w) <= 160


def test_extract_from_directory(sample_image_dir, tmp_path):
    paths = extract_frames_from_directory(sample_image_dir, max_frames=10)

    assert len(paths) == 10
    for p in paths:
        assert Path(p).exists()


def test_extract_from_directory_resize(sample_image_dir, tmp_path):
    paths = extract_frames_from_directory(
        sample_image_dir, max_frames=5, resize_long_edge=160
    )

    assert len(paths) == 5
    img = cv2.imread(paths[0])
    h, w = img.shape[:2]
    assert max(h, w) <= 160


def test_invalid_video(tmp_path):
    with pytest.raises(ValueError):
        extract_frames("/nonexistent/video.mp4", str(tmp_path / "out"))
