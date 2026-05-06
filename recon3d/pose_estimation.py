"""Camera pose estimation using VGGT or MASt3R."""

import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class PoseEstimationResult:
    """Result from pose estimation."""
    extrinsics: np.ndarray      # (N, 4, 4) world-to-camera matrices
    intrinsics: np.ndarray      # (N, 3, 3) camera intrinsic matrices
    point_cloud: np.ndarray     # (M, 3) 3D point positions
    point_colors: np.ndarray    # (M, 3) RGB colors in [0, 1]
    depth_maps: list[np.ndarray]  # list of (H, W) depth maps
    image_sizes: list[tuple[int, int]]  # (H, W) for each image
    is_metric: bool             # whether the scale is metric


def estimate_poses_vggt(
    image_paths: list[str],
    device: str = "cuda",
    conf_threshold: float = 1.5,
    max_batch_frames: int = 40,
) -> PoseEstimationResult:
    """Estimate camera poses and 3D structure using VGGT.

    VGGT outputs relative scale (not metric). Use metric_alignment to get metric scale.

    Args:
        image_paths: List of paths to input images.
        device: Device to run on.
        conf_threshold: Confidence threshold for point cloud filtering.
        max_batch_frames: Max frames per batch (controls GPU memory).

    Returns:
        PoseEstimationResult with camera poses and point cloud.
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    n_images = len(image_paths)
    all_extrinsics = []
    all_intrinsics = []
    all_depths = []
    all_points = []
    all_colors = []
    all_sizes = []

    for batch_start in range(0, n_images, max_batch_frames):
        batch_end = min(batch_start + max_batch_frames, n_images)
        batch_paths = image_paths[batch_start:batch_end]

        print(f"Processing frames {batch_start}-{batch_end} of {n_images}...")
        images = load_and_preprocess_images(batch_paths).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images)

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]

        ext_np = extrinsic.squeeze(0).cpu().numpy()
        int_np = intrinsic.squeeze(0).cpu().numpy()

        extrinsics_4x4 = np.zeros((ext_np.shape[0], 4, 4))
        extrinsics_4x4[:, :3, :] = ext_np
        extrinsics_4x4[:, 3, 3] = 1.0

        all_extrinsics.append(extrinsics_4x4)
        all_intrinsics.append(int_np)

        world_pts = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )

        conf = depth_conf.squeeze(0).cpu().numpy()
        imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(len(batch_paths)):
            h, w = world_pts[i].shape[:2]
            all_sizes.append((h, w))
            all_depths.append(depth[0, i, :, :, 0].cpu().numpy())

            pts = world_pts[i].reshape(-1, 3)
            c = conf[i].reshape(-1)
            colors = imgs_np[i].reshape(-1, 3)

            valid = c > conf_threshold
            all_points.append(pts[valid])
            all_colors.append(colors[valid])

        del predictions, images, depth, depth_conf
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    return PoseEstimationResult(
        extrinsics=np.concatenate(all_extrinsics, axis=0),
        intrinsics=np.concatenate(all_intrinsics, axis=0),
        point_cloud=np.concatenate(all_points, axis=0).astype(np.float32),
        point_colors=np.concatenate(all_colors, axis=0).astype(np.float32),
        depth_maps=all_depths,
        image_sizes=all_sizes,
        is_metric=False,
    )


def estimate_poses_mast3r(
    image_paths: list[str],
    device: str = "cuda",
    conf_threshold: float = 1.5,
    scene_graph: str = "swin-5",
    niter1: int = 300,
    niter2: int = 300,
    cache_dir: Optional[str] = None,
) -> PoseEstimationResult:
    """Estimate camera poses and 3D structure using MASt3R sparse global alignment.

    MASt3R outputs metric scale.

    Args:
        image_paths: List of paths to input images.
        device: Device to run on.
        conf_threshold: Confidence threshold for point cloud filtering.
        scene_graph: Scene graph type ('swin-K', 'logwin-K', 'complete').
        niter1: Iterations for coarse alignment.
        niter2: Iterations for fine alignment.
        cache_dir: Cache directory for MASt3R intermediate results.

    Returns:
        PoseEstimationResult with camera poses and point cloud.
    """
    import tempfile
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images

    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="recon3d_mast3r_")

    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    images = load_images(image_paths, size=512)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

    print(f"Running sparse global alignment ({len(pairs)} pairs)...")
    scene = sparse_global_alignment(
        image_paths, pairs, cache_dir, model,
        lr1=0.07, niter1=niter1,
        lr2=0.01, niter2=niter2,
        device=device,
        opt_depth=True,
        shared_intrinsics=True,
    )

    poses_c2w = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()

    pts3d_dense, depthmaps_dense, confs = scene.get_dense_pts3d(clean_depth=True)

    all_points = []
    all_colors = []
    all_depths = []
    all_sizes = []

    n = len(image_paths)
    intrinsics = np.zeros((n, 3, 3))
    extrinsics = np.zeros((n, 4, 4))

    for i in range(n):
        pts = pts3d_dense[i].detach().cpu().numpy().reshape(-1, 3)
        conf = confs[i].detach().cpu().numpy().reshape(-1)
        colors = scene.imgs[i].reshape(-1, 3)

        valid = conf > conf_threshold
        all_points.append(pts[valid])
        all_colors.append(colors[valid])

        h, w = pts3d_dense[i].shape[:2]
        all_sizes.append((h, w))
        all_depths.append(depthmaps_dense[i].detach().cpu().numpy())

        intrinsics[i] = np.array([
            [focals[i], 0, w / 2],
            [0, focals[i], h / 2],
            [0, 0, 1],
        ])
        extrinsics[i] = np.linalg.inv(poses_c2w[i])

    del model, scene
    torch.cuda.empty_cache()

    return PoseEstimationResult(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=np.concatenate(all_points, axis=0).astype(np.float32),
        point_colors=np.concatenate(all_colors, axis=0).astype(np.float32),
        depth_maps=all_depths,
        image_sizes=all_sizes,
        is_metric=True,
    )
