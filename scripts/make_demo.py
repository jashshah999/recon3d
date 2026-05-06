"""Generate a demo GIF: create synthetic scene -> reconstruct -> render novel views."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path


def create_rich_scene(output_dir: Path, n_images: int = 12):
    """Create a visually interesting synthetic multi-view scene."""
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = 512, 512
    fx = fy = 350.0
    cx, cy = W / 2, H / 2

    np.random.seed(42)

    points_list = []
    colors_list = []

    # Sphere of points (red-orange)
    for _ in range(2000):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = 0.4 + np.random.randn() * 0.02
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) - 0.2
        z = r * np.cos(phi)
        points_list.append([x, y, z])
        t = phi / np.pi
        colors_list.append([0.9, 0.3 + 0.4 * t, 0.1])

    # Ground plane (green-brown)
    for _ in range(3000):
        x = np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(-1.5, 1.5)
        y = 0.4 + np.random.randn() * 0.01
        points_list.append([x, y, z])
        d = np.sqrt(x**2 + z**2)
        colors_list.append([0.2 + 0.1 * np.sin(x * 5), 0.5 - 0.1 * d, 0.15])

    # Vertical pillar (blue)
    for _ in range(1500):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 0.1 + np.random.randn() * 0.01
        x = r * np.cos(angle) + 0.7
        z = r * np.sin(angle) + 0.3
        y = np.random.uniform(-0.8, 0.4)
        points_list.append([x, y, z])
        colors_list.append([0.1, 0.2, 0.7 + 0.2 * (y + 0.8) / 1.2])

    # Second pillar (cyan)
    for _ in range(1500):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 0.12 + np.random.randn() * 0.01
        x = r * np.cos(angle) - 0.8
        z = r * np.sin(angle) - 0.4
        y = np.random.uniform(-0.6, 0.4)
        points_list.append([x, y, z])
        colors_list.append([0.1, 0.6 + 0.2 * (y + 0.6) / 1.0, 0.7])

    # Floating ring (yellow)
    for _ in range(1000):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 0.6 + np.random.randn() * 0.03
        x = r * np.cos(angle)
        z = r * np.sin(angle)
        y = -0.5 + np.random.randn() * 0.02
        points_list.append([x, y, z])
        colors_list.append([0.9, 0.8, 0.1])

    points = np.array(points_list, dtype=np.float32)
    colors = np.array(colors_list, dtype=np.float32)
    colors = np.clip(colors, 0, 1)

    image_paths = []
    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 3.0
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.3 + 0.4 * np.sin(angle * 0.5),
            radius * np.sin(angle),
        ])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        pts_cam = (R @ points.T).T + t
        z = pts_cam[:, 2]
        valid = z > 0.1

        u = (fx * pts_cam[:, 0] / z + cx).astype(int)
        v = (fy * pts_cam[:, 1] / z + cy).astype(int)

        # Sort by depth (back to front) for proper occlusion
        order = np.argsort(-z)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Gradient background
        for row in range(H):
            t = row / H
            img[row, :] = [int(20 + 15 * t), int(20 + 10 * t), int(30 + 20 * t)]

        for j in order:
            if valid[j] and 0 <= u[j] < W and 0 <= v[j] < H:
                c = (colors[j] * 255).astype(np.uint8)
                depth_scale = max(2, int(6 - z[j] * 1.5))
                cv2.circle(img, (u[j], v[j]), depth_scale, c.tolist(), -1)

        path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_paths.append(str(path))

    return image_paths, points, colors


def render_novel_views(
    ply_path: str,
    checkpoint_path: str,
    output_dir: Path,
    n_views: int = 60,
    H: int = 512,
    W: int = 512,
):
    """Render a spinning camera around the reconstructed scene."""
    from gsplat.rendering import rasterization

    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    fx = fy = 350.0
    cx, cy = W / 2, H / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device="cuda")

    frames = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        radius = 3.0
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.3 + 0.3 * np.sin(angle * 0.7),
            radius * np.sin(angle),
        ])

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        viewmat = torch.from_numpy(w2c).cuda()[None]

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        with torch.no_grad():
            renders, _, _ = rasterization(
                means=splats["means"],
                quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh,
                viewmats=viewmat,
                Ks=K[None],
                width=W,
                height=H,
                sh_degree=3,
                near_plane=0.01,
                far_plane=100.0,
            )

        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        frame_path = output_dir / f"render_{i:04d}.png"
        cv2.imwrite(str(frame_path), img_bgr)
        frames.append(img)

    return frames


def frames_to_gif(frames, output_path, fps=20):
    """Convert frames to a GIF."""
    from PIL import Image

    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )


def make_side_by_side(input_frames_dir, render_frames, output_path, fps=20):
    """Create a side-by-side GIF: input views | novel renders."""
    from PIL import Image

    input_paths = sorted(Path(input_frames_dir).glob("frame_*.jpg"))
    n_renders = len(render_frames)

    combined_frames = []
    for i in range(n_renders):
        # Pick corresponding input frame (cycle through)
        input_idx = i % len(input_paths)
        input_img = cv2.cvtColor(cv2.imread(str(input_paths[input_idx])), cv2.COLOR_BGR2RGB)

        render_img = render_frames[i]

        # Resize both to same height
        h = min(input_img.shape[0], render_img.shape[0], 384)
        w_in = int(input_img.shape[1] * h / input_img.shape[0])
        w_re = int(render_img.shape[1] * h / render_img.shape[0])

        input_resized = cv2.resize(input_img, (w_in, h))
        render_resized = cv2.resize(render_img, (w_re, h))

        # Add labels
        label_h = 30
        input_labeled = np.zeros((h + label_h, w_in, 3), dtype=np.uint8)
        input_labeled[label_h:] = input_resized
        input_labeled[:label_h] = [30, 30, 30]
        cv2.putText(input_labeled, "Input Views", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        render_labeled = np.zeros((h + label_h, w_re, 3), dtype=np.uint8)
        render_labeled[label_h:] = render_resized
        render_labeled[:label_h] = [30, 30, 30]
        cv2.putText(render_labeled, "Novel View (recon3d)", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Separator
        sep = np.ones((h + label_h, 4, 3), dtype=np.uint8) * 80

        combined = np.concatenate([input_labeled, sep, render_labeled], axis=1)
        combined_frames.append(combined)

    pil_frames = [Image.fromarray(f) for f in combined_frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    base_dir = Path("/tmp/recon3d_demo")
    input_dir = base_dir / "input_frames"
    recon_dir = base_dir / "reconstruction"
    render_dir = base_dir / "renders"

    print("=" * 60)
    print("Step 1: Creating synthetic multi-view scene")
    print("=" * 60)
    image_paths, points, colors = create_rich_scene(input_dir, n_images=12)
    print(f"Created {len(image_paths)} input images")

    print("\n" + "=" * 60)
    print("Step 2: Running recon3d pipeline")
    print("=" * 60)
    from recon3d.pipeline import reconstruct, PipelineConfig
    from recon3d.gaussian_train import TrainConfig

    config = PipelineConfig(
        pose_method="vggt",
        max_frames=12,
        metric_align=False,
        train_config=TrainConfig(max_steps=3000, log_every=500),
        launch_viewer=False,
        device="cuda",
    )
    ply_path = reconstruct(str(input_dir), str(recon_dir), config)

    print("\n" + "=" * 60)
    print("Step 3: Rendering novel views")
    print("=" * 60)
    render_frames = render_novel_views(
        ply_path,
        str(recon_dir / "checkpoint.pt"),
        render_dir,
        n_views=60,
    )
    print(f"Rendered {len(render_frames)} novel views")

    print("\n" + "=" * 60)
    print("Step 4: Creating GIF")
    print("=" * 60)
    gif_path = base_dir / "demo.gif"
    make_side_by_side(str(input_dir), render_frames, str(gif_path), fps=15)

    # Also make render-only GIF
    render_gif = base_dir / "render_only.gif"
    frames_to_gif(render_frames, str(render_gif), fps=15)
    print(f"Render-only GIF: {render_gif} ({os.path.getsize(str(render_gif)) / 1024:.0f} KB)")

    print("\nDone!")
