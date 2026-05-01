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
@click.option("--no-metric", is_flag=True, default=False,
              help="Skip metric scale alignment (faster, but no real-world scale).")
@click.option("--no-viewer", is_flag=True, default=False,
              help="Don't launch the interactive viewer after reconstruction.")
@click.option("--device", type=str, default="cuda",
              help="Device to use (cuda or cpu).")
def run(input_path, output_dir, pose_method, max_frames, fps, steps,
        resize, no_metric, no_viewer, device):
    """Reconstruct a 3D scene from a video or image directory.

    INPUT_PATH can be a video file (.mp4, .mov, etc.) or a directory of images.

    Examples:

        recon3d run my_video.mp4

        recon3d run my_video.mp4 -o output/ --steps 15000

        recon3d run ./images/ --pose-method mast3r

        recon3d run my_video.mp4 --no-metric --max-frames 50
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
        metric_align=not no_metric,
        train_config=TrainConfig(max_steps=steps),
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


if __name__ == "__main__":
    main()
