# recon3d

**One-command 3D reconstruction from video. No COLMAP needed.**

Turn any phone video or set of images into a 3D Gaussian Splat — with metric scale, in minutes, on a single GPU.

```bash
pip install recon3d
recon3d run my_video.mp4
```

https://github.com/user-attachments/assets/placeholder-demo.gif

## Why?

Every 3D Gaussian Splatting pipeline today requires COLMAP for camera pose estimation. COLMAP is slow, fragile, fails on textureless scenes, and is painful to install. Meanwhile, feed-forward models like [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025 Best Paper) can estimate poses in seconds — but nobody has wired them into a usable reconstruction pipeline.

**recon3d** connects the dots:

```
Video/Images → VGGT (poses) → MoGe-2 (metric scale) → gsplat (training) → .ply + viewer
```

No COLMAP. No CUDA compilation nightmares. One pip install. One command.

## Features

- **No COLMAP** — Uses VGGT or MASt3R for camera pose estimation
- **Metric scale** — Real-world units via MoGe-2 alignment
- **One command** — Video in, Gaussian Splat out
- **Fast** — Minutes, not hours. VGGT processes 100 frames in ~3 seconds on an H100
- **Interactive viewer** — Built-in web viewer via viser
- **Multiple exports** — .ply (standard), .splat (web viewers), checkpoints

## Quick Start

### Install

```bash
# Core package
pip install recon3d

# Install model backends (pick what you need)
pip install recon3d[vggt]    # VGGT for pose estimation (recommended)
pip install recon3d[moge]    # MoGe-2 for metric scale
pip install recon3d[gsplat]  # gsplat for Gaussian Splatting training
pip install recon3d[all]     # everything
```

Or from source:

```bash
git clone https://github.com/jashshah999/recon3d.git
cd recon3d
pip install -e ".[all]"
```

### Reconstruct from video

```bash
recon3d run my_video.mp4
```

### Reconstruct from images

```bash
recon3d run ./my_images/
```

### View a reconstruction

```bash
recon3d view output/scene.ply
```

## Options

```
recon3d run INPUT_PATH [OPTIONS]

Options:
  -o, --output DIR          Output directory (default: ./recon3d_output/<name>)
  --pose-method [vggt|mast3r]  Pose estimation method (default: vggt)
  --max-frames INT          Max frames to extract (default: 80)
  --fps FLOAT               Target FPS for extraction (default: 2.0)
  --steps INT               Gaussian Splatting training steps (default: 7000)
  --resize INT              Resize long edge in pixels (default: 960)
  --no-metric               Skip metric alignment (faster)
  --no-viewer               Don't launch viewer after reconstruction
  --device TEXT             Device: cuda or cpu (default: cuda)
```

## Pipeline

### Step 1: Frame Extraction
Extracts frames from video at target FPS, filters blurry frames using Laplacian variance, resizes to target resolution.

### Step 2: Pose Estimation
**VGGT** (default): Feed-forward transformer that jointly predicts camera poses, depth maps, and 3D point clouds from unordered images. No optimization loop, no feature matching — just a single forward pass. Relative scale.

**MASt3R** (alternative): Sparse global alignment over pairwise predictions. Metric scale natively, but slower and NC-licensed.

### Step 3: Metric Alignment
When using VGGT (which outputs relative scale), recon3d runs [MoGe-2](https://github.com/microsoft/MoGe) on a subset of frames to estimate metric depth, then computes a global scale factor to convert the entire reconstruction to real-world meters.

### Step 4: Gaussian Splatting Training
Trains 3D Gaussian Splatting using [gsplat](https://github.com/nerfstudio-project/gsplat) with the estimated poses and initial point cloud. Exports to .ply and .splat formats.

## Output Structure

```
recon3d_output/my_video/
├── frames/          # Extracted video frames
├── scene.ply        # Gaussian Splat (standard PLY)
├── scene.splat      # Gaussian Splat (web viewer format)
└── checkpoint.pt    # Training checkpoint
```

## Python API

```python
from recon3d.pipeline import reconstruct, PipelineConfig
from recon3d.gaussian_train import TrainConfig

config = PipelineConfig(
    pose_method="vggt",
    max_frames=100,
    train_config=TrainConfig(max_steps=15000),
)

ply_path = reconstruct("my_video.mp4", "output/", config)
```

Use individual components:

```python
from recon3d.video import extract_frames
from recon3d.pose_estimation import estimate_poses_vggt
from recon3d.metric_align import align_to_metric
from recon3d.gaussian_train import train_gaussians

# Extract
frames = extract_frames("video.mp4", "frames/", max_frames=80)

# Estimate poses
result = estimate_poses_vggt(frames)

# Align to metric
ext, intr, pts, depths = align_to_metric(
    frames, result.extrinsics, result.intrinsics,
    result.point_cloud, result.depth_maps
)

# Train
train_gaussians(frames, ext, intr, pts, result.point_colors, "output/")
```

## Model Weights

Models are downloaded automatically from HuggingFace on first run:

| Model | Size | License | Used For |
|-------|------|---------|----------|
| [VGGT-1B](https://huggingface.co/facebook/VGGT-1B) | 1B params | Meta Research | Pose estimation |
| [MoGe-2-ViT-L](https://huggingface.co/Ruicheng/moge-2-vitl) | 326M params | MIT | Metric alignment |
| [MASt3R-ViT-L](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric) | ~1B params | CC BY-NC-SA 4.0 | Alt. pose estimation |

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM (with `--max-frames 30 --resize 512`)
- **Recommended**: NVIDIA GPU with 24GB VRAM (RTX 3090/4090)
- **For 100+ frames**: 40GB+ VRAM (A100) or reduce `--max-frames`

## Limitations

- VGGT outputs relative scale — metric alignment via MoGe-2 is approximate (typically within 10-20% of ground truth)
- Quality may be slightly lower than COLMAP-based pipelines on some scenes (~1-2 dB PSNR), but 10-100x faster
- Dynamic objects in the scene will cause artifacts
- Very large scenes (building-scale) may need frame subsampling

## Acknowledgments

Built on the shoulders of giants:
- [VGGT](https://github.com/facebookresearch/vggt) — Meta (CVPR 2025 Best Paper)
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Nerfstudio team
- [MoGe-2](https://github.com/microsoft/MoGe) — Microsoft
- [MASt3R](https://github.com/naver/mast3r) — Naver Labs
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) — INRIA

## License

MIT

## Citation

```bibtex
@software{recon3d,
  author = {Shah, Jash},
  title = {recon3d: One-command 3D reconstruction from video without COLMAP},
  year = {2026},
  url = {https://github.com/jashshah999/recon3d},
}
```
