"""Align VGGT's relative-scale output to metric scale using MoGe-2."""

import numpy as np
import torch
import cv2
from typing import Optional


def align_to_metric(
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    point_cloud: np.ndarray,
    depth_maps: list[np.ndarray],
    device: str = "cuda",
    n_reference_frames: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Align relative-scale reconstruction to metric scale using MoGe-2.

    Uses MoGe-2 metric depth on a subset of frames, then computes a global
    scale factor to convert the entire reconstruction to metric.

    Args:
        image_paths: Paths to input images.
        extrinsics: (N, 4, 4) world-to-camera matrices.
        intrinsics: (N, 3, 3) camera intrinsic matrices.
        point_cloud: (M, 3) point cloud in relative scale.
        depth_maps: List of (H, W) depth maps in relative scale.
        device: Device to run MoGe-2 on.
        n_reference_frames: Number of frames to use for scale estimation.

    Returns:
        Tuple of (extrinsics, intrinsics, point_cloud, depth_maps) in metric scale.
    """
    from moge.model.v2 import MoGeModel

    print("Loading MoGe-2 for metric alignment...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)

    n_frames = len(image_paths)
    ref_indices = np.linspace(0, n_frames - 1, min(n_reference_frames, n_frames), dtype=int)

    scale_ratios = []

    for idx in ref_indices:
        image = cv2.cvtColor(cv2.imread(image_paths[idx]), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1)

        with torch.no_grad():
            output = model.infer(image_tensor)

        metric_depth = output["depth"].cpu().numpy()

        rel_depth = depth_maps[idx]

        if metric_depth.shape != rel_depth.shape:
            metric_depth = cv2.resize(
                metric_depth, (rel_depth.shape[1], rel_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        valid = (rel_depth > 1e-6) & (metric_depth > 1e-6)
        if valid.sum() < 100:
            continue

        ratio = np.median(metric_depth[valid] / rel_depth[valid])
        if 0.01 < ratio < 1000:
            scale_ratios.append(ratio)

    del model
    torch.cuda.empty_cache()

    if not scale_ratios:
        print("WARNING: Could not estimate metric scale. Using relative scale.")
        return extrinsics, intrinsics, point_cloud, depth_maps

    scale = np.median(scale_ratios)
    print(f"Metric scale factor: {scale:.4f}")

    scaled_extrinsics = extrinsics.copy()
    scaled_extrinsics[:, :3, 3] *= scale

    scaled_points = point_cloud * scale
    scaled_depths = [d * scale for d in depth_maps]

    return scaled_extrinsics, intrinsics, scaled_points, scaled_depths
