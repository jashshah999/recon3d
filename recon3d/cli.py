"""Command-line interface for recon3d."""

import click
from pathlib import Path


@click.group()
@click.version_option()
def main():
    """recon3d: One-command 3D reconstruction from video. No COLMAP needed."""
    pass


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", "output_dir", type=click.Path(), default=None,
              help="Output directory. Defaults to ./recon3d_output/")
@click.option("--pose-method", type=click.Choice(["vggt", "mast3r"]), default="vggt",
              help="Pose estimation method.")
@click.option("--max-frames", type=int, default=80,
              help="Maximum frames to extract from video.")
@click.option("--fps", type=float, default=2.0,
              help="Target FPS for frame extraction.")
@click.option("--steps", type=int, default=7000,
              help="Number of Gaussian Splatting training steps.")
@click.option("--resize", type=int, default=960,
              help="Resize long edge to this many pixels.")
@click.option("--chunk-size", type=int, default=20,
              help="Frames per VGGT chunk for long sequences.")
@click.option("--no-metric", is_flag=True, default=False,
              help="Skip metric scale alignment (faster, but no real-world scale).")
@click.option("--no-viewer", is_flag=True, default=False,
              help="Don't launch the interactive viewer after reconstruction.")
@click.option("--no-factor-graph", is_flag=True, default=False,
              help="Disable factor graph refinement for chunked sequences.")
@click.option("--isam2/--batch-lm", default=True,
              help="Use iSAM2 (incremental) vs batch Levenberg-Marquardt for factor graph.")
@click.option("--loop-threshold", type=float, default=0.65,
              help="DINOv2 similarity threshold for loop closure detection (0-1).")
@click.option("--robust-kernel", type=click.Choice(["cauchy", "huber", "none"]), default="cauchy",
              help="Robust kernel for outlier rejection in factor graph.")
@click.option("--mesh", is_flag=True, default=False,
              help="Export a triangle mesh (.obj) via TSDF fusion.")
@click.option("--device", type=str, default="cuda",
              help="Device to use (cuda or cpu).")
def run(input_path, output_dir, pose_method, max_frames, fps, steps,
        resize, chunk_size, no_metric, no_viewer, no_factor_graph, isam2,
        loop_threshold, robust_kernel, mesh, device):
    """Reconstruct a 3D scene from a video or image directory.

    INPUT_PATH can be a video file (.mp4, .mov, etc.) or a directory of images.

    Examples:

        recon3d run my_video.mp4

        recon3d run my_video.mp4 -o output/ --steps 15000

        recon3d run ./images/ --pose-method mast3r

        recon3d run my_video.mp4 --no-metric --max-frames 50

        recon3d run long_video.mp4 --max-frames 200 --mesh
    """
    from .pipeline import reconstruct, PipelineConfig
    from .gaussian_train import TrainConfig

    if output_dir is None:
        input_name = Path(input_path).stem
        output_dir = f"./recon3d_output/{input_name}"

    config = PipelineConfig(
        max_frames=max_frames,
        target_fps=fps,
        resize_long_edge=resize,
        pose_method=pose_method,
        chunk_size=chunk_size,
        use_factor_graph=not no_factor_graph,
        use_isam2=isam2,
        loop_closure_threshold=loop_threshold,
        robust_kernel=robust_kernel,
        metric_align=not no_metric,
        train_config=TrainConfig(max_steps=steps),
        export_mesh=mesh,
        launch_viewer=not no_viewer,
        device=device,
    )

    print(f"""
recon3d v0.1.0
==============
Input:        {input_path}
Output:       {output_dir}
Pose method:  {pose_method}
Max frames:   {max_frames}
Train steps:  {steps}
Metric align: {not no_metric}
Factor graph: {not no_factor_graph}
Mesh export:  {mesh}
Device:       {device}
""")

    reconstruct(input_path, output_dir, config)


@main.command()
@click.argument("ply_path", type=click.Path(exists=True))
@click.option("--host", type=str, default="0.0.0.0")
@click.option("--port", type=int, default=8080)
def view(ply_path, host, port):
    """Launch the interactive viewer for a reconstructed scene.

    Examples:

        recon3d view output/scene.ply

        recon3d view output/scene.ply --port 9090
    """
    from .viewer import launch_viewer
    launch_viewer(ply_path, host=host, port=port)


@main.command()
def check():
    """Check system dependencies, GPU info, and installed backends."""
    import sys
    print("recon3d check")
    print(f"{'='*40}")
    print(f"Python: {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            vram_gb = vram_bytes / 1e9
            print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb < 8:
                print("  WARNING: <8GB VRAM. Use --resize 640 --max-frames 30")
            elif vram_gb < 16:
                print("  Recommended: --resize 720 --max-frames 50")
            elif vram_gb < 24:
                print("  Recommended: --resize 960 --max-frames 80")
            else:
                print("  Full resolution supported")
        else:
            print("GPU: None (CPU only, will be very slow)")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # Backends
    backends = {
        "VGGT": "vggt",
        "MASt3R": "mast3r",
        "MoGe-2": "moge",
        "gsplat": "gsplat",
        "GTSAM": "gtsam",
        "Open3D": "open3d",
        "viser": "viser",
    }
    print("\nBackends:")
    for name, module in backends.items():
        try:
            __import__(module)
            print(f"  {name}: installed")
        except ImportError:
            print(f"  {name}: not installed")


if __name__ == "__main__":
    main()
