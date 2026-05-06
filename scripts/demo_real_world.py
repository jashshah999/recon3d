"""Run recon3d on real-world images and create comparison GIFs."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path
from PIL import Image


def run_and_render(scene_name, image_dir, output_base, max_frames=20, train_steps=10000):
    """Run full pipeline + render novel views + create GIF."""
    output_dir = output_base / scene_name
    render_dir = output_dir / "renders"

    print(f"\n{'='*60}")
    print(f"Processing: {scene_name}")
    print(f"{'='*60}")

    from recon3d.pipeline import reconstruct, PipelineConfig
    from recon3d.gaussian_train import TrainConfig

    config = PipelineConfig(
        pose_method="vggt",
        max_frames=max_frames,
        metric_align=False,
        train_config=TrainConfig(max_steps=train_steps, log_every=1000),
        launch_viewer=False,
        resize_long_edge=706,
    )

    ply_path = reconstruct(str(image_dir), str(output_dir), config)

    # Load checkpoint and render
    print("Rendering from training views...")
    from gsplat.rendering import rasterization
    from recon3d.pose_estimation import estimate_poses_vggt

    ckpt = torch.load(str(output_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    # Get the poses that were used
    image_paths = sorted([str(p) for p in Path(image_dir).glob("*.png")])
    if not image_paths:
        image_paths = sorted([str(p) for p in Path(image_dir).glob("*.jpg")])
    if len(image_paths) > max_frames:
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]

    result = estimate_poses_vggt(image_paths, device="cuda", conf_threshold=1.0)

    img0 = cv2.imread(image_paths[0])
    H, W = img0.shape[:2]
    K = torch.from_numpy(result.intrinsics[0].astype(np.float32)).cuda()

    # Render from training viewpoints
    gt_images = []
    rendered_images = []
    for i in range(len(image_paths)):
        gt = cv2.cvtColor(cv2.imread(image_paths[i]), cv2.COLOR_BGR2RGB)
        gt_images.append(gt)

        viewmat = torch.from_numpy(result.extrinsics[i].astype(np.float32)).cuda()[None]
        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh, viewmats=viewmat, Ks=K[None],
                width=W, height=H, sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
        rendered = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        rendered_images.append(rendered)

    # Render novel views (orbit)
    print("Rendering novel views...")
    pts = result.point_cloud
    center = np.median(pts, axis=0)
    cam_positions = []
    for i in range(len(result.extrinsics)):
        c2w = np.linalg.inv(result.extrinsics[i])
        cam_positions.append(c2w[:3, 3])
    cam_positions = np.array(cam_positions)
    avg_dist = np.mean(np.linalg.norm(cam_positions - center, axis=1))

    # Estimate the "up" direction from camera poses
    up_dirs = []
    for i in range(len(result.extrinsics)):
        c2w = np.linalg.inv(result.extrinsics[i])
        up_dirs.append(c2w[:3, 1])  # y-axis of camera
    avg_up = np.mean(up_dirs, axis=0)
    avg_up /= np.linalg.norm(avg_up)

    novel_frames = []
    n_novel = 60
    for i in range(n_novel):
        angle = 2 * np.pi * i / n_novel
        # Orbit in a plane perpendicular to avg_up
        # Find two basis vectors in that plane
        arbitrary = np.array([1, 0, 0]) if abs(avg_up[0]) < 0.9 else np.array([0, 1, 0])
        basis1 = np.cross(avg_up, arbitrary)
        basis1 /= np.linalg.norm(basis1)
        basis2 = np.cross(avg_up, basis1)

        cam_pos = center + avg_dist * (np.cos(angle) * basis1 + np.sin(angle) * basis2)
        cam_pos += avg_up * avg_dist * 0.1 * np.sin(angle * 0.5)

        forward = center - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, -avg_up)
        norm_r = np.linalg.norm(right)
        if norm_r < 1e-6:
            right = np.cross(forward, avg_up + np.array([0.01, 0, 0]))
            norm_r = np.linalg.norm(right)
        right /= norm_r
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
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh, viewmats=viewmat, Ks=K[None],
                width=W, height=H, sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        novel_frames.append(img)

    # Create GIFs
    print("Creating GIFs...")
    render_dir = output_dir / "gifs"
    render_dir.mkdir(parents=True, exist_ok=True)

    # 1. GT vs Rendered comparison (cycling through training views)
    target_h = 400
    combined = []
    for i in range(len(gt_images)):
        gt = gt_images[i]
        ren = rendered_images[i]

        # Resize maintaining aspect ratio
        scale = target_h / gt.shape[0]
        new_w = int(gt.shape[1] * scale)
        gt_r = cv2.resize(gt, (new_w, target_h))
        ren_r = cv2.resize(ren, (new_w, target_h))

        lh = 30
        def label(img, text):
            out = np.zeros((target_h + lh, new_w, 3), dtype=np.uint8)
            out[:lh] = 25
            out[lh:] = img
            cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            return out

        sep = np.ones((target_h + lh, 3, 3), dtype=np.uint8) * 50
        frame = np.concatenate([label(gt_r, "Ground Truth"), sep, label(ren_r, "recon3d")], axis=1)
        combined.append(frame)

    pil = [Image.fromarray(f) for f in combined]
    comp_path = str(render_dir / "comparison.gif")
    pil[0].save(comp_path, save_all=True, append_images=pil[1:], duration=400, loop=0, optimize=True)
    print(f"  Comparison: {comp_path} ({os.path.getsize(comp_path)/1024:.0f} KB)")

    # 2. Novel view orbit
    pil_novel = [Image.fromarray(cv2.resize(f, (512, int(512 * H / W)))) for f in novel_frames]
    novel_path = str(render_dir / "novel_views.gif")
    pil_novel[0].save(novel_path, save_all=True, append_images=pil_novel[1:], duration=66, loop=0, optimize=True)
    print(f"  Novel views: {novel_path} ({os.path.getsize(novel_path)/1024:.0f} KB)")

    del splats
    torch.cuda.empty_cache()

    return comp_path, novel_path


if __name__ == "__main__":
    output_base = Path("/tmp/recon3d_real")

    # Run on LLFF Flower (real photos)
    comp, novel = run_and_render(
        "flower",
        "/home/ubuntu/vggt/examples/llff_flower/images",
        output_base,
        max_frames=15,
        train_steps=10000,
    )
    print(f"\nFlower done: {comp}, {novel}")
