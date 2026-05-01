"""Extract frames from video files."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def extract_frames(
    video_path: str,
    output_dir: str,
    max_frames: int = 100,
    target_fps: Optional[float] = None,
    min_blur_score: float = 50.0,
    resize_long_edge: Optional[int] = None,
) -> list[str]:
    """Extract frames from a video, filtering blurry ones.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to save extracted frames.
        max_frames: Maximum number of frames to extract.
        target_fps: Target FPS for extraction. None = use video FPS.
        min_blur_score: Minimum Laplacian variance to keep a frame.
        resize_long_edge: Resize frames so the long edge is this many pixels.

    Returns:
        List of paths to extracted frame images.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_fps is None or target_fps >= video_fps:
        frame_interval = 1
    else:
        frame_interval = int(round(video_fps / target_fps))

    candidate_indices = list(range(0, total_frames, frame_interval))

    if len(candidate_indices) > max_frames * 3:
        step = len(candidate_indices) // (max_frames * 3)
        candidate_indices = candidate_indices[::step]

    frames_with_scores = []

    for idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < min_blur_score:
            continue

        frames_with_scores.append((idx, blur_score, frame))

    cap.release()

    frames_with_scores.sort(key=lambda x: x[1], reverse=True)

    if len(frames_with_scores) > max_frames:
        frames_with_scores = frames_with_scores[:max_frames]

    frames_with_scores.sort(key=lambda x: x[0])

    saved_paths = []
    for i, (idx, _, frame) in enumerate(frames_with_scores):
        if resize_long_edge is not None:
            h, w = frame.shape[:2]
            long_edge = max(h, w)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / long_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(str(frame_path))

    return saved_paths


def extract_frames_from_directory(
    image_dir: str,
    max_frames: int = 100,
    resize_long_edge: Optional[int] = None,
) -> list[str]:
    """Load frames from a directory of images.

    Args:
        image_dir: Directory containing images.
        max_frames: Maximum number of frames to use.
        resize_long_edge: Resize frames so the long edge is this many pixels.

    Returns:
        List of paths to frame images.
    """
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in extensions]
    )

    if len(image_paths) > max_frames:
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]

    if resize_long_edge is not None:
        resized_dir = image_dir / "_recon3d_resized"
        resized_dir.mkdir(exist_ok=True)
        resized_paths = []
        for p in image_paths:
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            long_edge = max(h, w)
            if long_edge > resize_long_edge:
                scale = resize_long_edge / long_edge
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            out_path = resized_dir / p.name
            cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            resized_paths.append(str(out_path))
        return resized_paths

    return [str(p) for p in image_paths]
