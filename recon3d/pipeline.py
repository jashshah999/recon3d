"""End-to-end reconstruction pipeline."""

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .video import extract_frames, extract_frames_from_directory
from .pose_estimation import estimate_poses_vggt, estimate_poses_mast3r
from .gaussian_train import train_gaussians, TrainConfig


@dataclass
class PipelineConfig:
    # Input
    max_frames: int = 80
    target_fps: Optional[float] = 2.0
    resize_long_edge: int = 960
    min_blur_score: float = 50.0

    # Pose estimation
    pose_method: str = "vggt"  # "vggt" or "mast3r"
    pose_conf_threshold: float = 1.0

    # Metric alignment (only for VGGT which outputs relative scale)
    metric_align: bool = True
    n_metric_reference_frames: int = 5

    # Gaussian splatting
    train_config: TrainConfig = field(default_factory=TrainConfig)

    # Viewer
    launch_viewer: bool = True

    device: str = "cuda"


def reconstruct(
    input_path: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
) -> str:
    """Run the full reconstruction pipeline.

    Args:
        input_path: Path to a video file or directory of images.
        output_dir: Directory to save all outputs.
        config: Pipeline configuration.

    Returns:
        Path to the exported .ply file.
    """
    if config is None:
        config = PipelineConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    t_start = time.time()

    # Step 1: Extract frames
    input_path = Path(input_path)
    print(f"\n{'='*60}")
    print(f"Step 1/4: Extracting frames")
    print(f"{'='*60}")

    if input_path.is_dir():
        image_paths = extract_frames_from_directory(
            str(input_path),
            max_frames=config.max_frames,
            resize_long_edge=config.resize_long_edge,
        )
    elif input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        image_paths = extract_frames(
            str(input_path),
            str(frames_dir),
            max_frames=config.max_frames,
            target_fps=config.target_fps,
            min_blur_score=config.min_blur_score,
            resize_long_edge=config.resize_long_edge,
        )
    else:
        raise ValueError(
            f"Input must be a video file or directory of images, got: {input_path}"
        )

    print(f"Extracted {len(image_paths)} frames")

    if len(image_paths) < 3:
        raise ValueError(f"Need at least 3 frames, got {len(image_paths)}")

    # Step 2: Estimate poses
    print(f"\n{'='*60}")
    print(f"Step 2/4: Estimating camera poses ({config.pose_method})")
    print(f"{'='*60}")

    if config.pose_method == "vggt":
        result = estimate_poses_vggt(
            image_paths,
            device=config.device,
            conf_threshold=config.pose_conf_threshold,
        )
    elif config.pose_method == "mast3r":
        result = estimate_poses_mast3r(
            image_paths,
            device=config.device,
            conf_threshold=config.pose_conf_threshold,
        )
    else:
        raise ValueError(f"Unknown pose method: {config.pose_method}")

    print(f"Got {len(result.extrinsics)} poses, {len(result.point_cloud)} points")

    if len(result.point_cloud) < 100:
        raise ValueError(
            f"Only {len(result.point_cloud)} points recovered — not enough for reconstruction. "
            "This usually means the input lacks sufficient 3D parallax (camera needs to move "
            "around the scene, not just pan). Try different input or lower --pose-conf-threshold."
        )

    extrinsics = result.extrinsics
    intrinsics = result.intrinsics
    point_cloud = result.point_cloud
    point_colors = result.point_colors
    depth_maps = result.depth_maps

    # Step 3: Metric alignment
    print(f"\n{'='*60}")
    print(f"Step 3/4: Metric scale alignment")
    print(f"{'='*60}")

    if config.metric_align and not result.is_metric:
        from .metric_align import align_to_metric
        extrinsics, intrinsics, point_cloud, depth_maps = align_to_metric(
            image_paths,
            extrinsics,
            intrinsics,
            point_cloud,
            depth_maps,
            device=config.device,
            n_reference_frames=config.n_metric_reference_frames,
        )
        print("Metric alignment complete")
    else:
        print("Skipping (already metric or disabled)")

    # Step 4: Train Gaussian Splatting
    print(f"\n{'='*60}")
    print(f"Step 4/4: Training Gaussian Splatting")
    print(f"{'='*60}")

    ply_path = train_gaussians(
        image_paths=image_paths,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=point_cloud,
        point_colors=point_colors,
        output_dir=str(output_dir),
        config=config.train_config,
        device=config.device,
    )

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Total time: {t_total:.1f}s")
    print(f"{'='*60}")
    print(f"Outputs:")
    print(f"  PLY:        {output_dir / 'scene.ply'}")
    print(f"  Splat:      {output_dir / 'scene.splat'}")
    print(f"  Checkpoint: {output_dir / 'checkpoint.pt'}")

    if config.launch_viewer:
        print(f"\nLaunching viewer...")
        try:
            from .viewer import launch_viewer
            launch_viewer(str(output_dir / "scene.ply"), image_paths, extrinsics, intrinsics)
        except Exception as e:
            print(f"Viewer failed: {e}")
            print("You can view the .ply file in any Gaussian Splatting viewer.")

    return ply_path
