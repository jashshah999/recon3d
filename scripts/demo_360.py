"""Run recon3d on a 360-degree dataset and create orbit GIF."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path
from PIL import Image


def main():
    image_dir = "/tmp/lego_rgb"
    output_dir = Path("/tmp/recon3d_lego")

    print("Running recon3d on lego (360-degree, 20 images)...")
    from recon3d.pipeline import reconstruct, PipelineConfig
    from recon3d.gaussian_train import TrainConfig

    config = PipelineConfig(
        pose_method="vggt",
        max_frames=20,
        metric_align=False,
        train_config=TrainConfig(max_steps=10000, log_every=1000, max_gaussians=200_000, init_points=120_000),
        launch_viewer=False,
        resize_long_edge=800,
    )
    ply_path = reconstruct(image_dir, str(output_dir), config)

    print("\nRendering novel views...")
    from gsplat.rendering import rasterization
    from recon3d.pose_estimation import estimate_poses_vggt

    ckpt = torch.load(str(output_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    image_paths = sorted([str(p) for p in Path(image_dir).glob("*.png")])
    result = estimate_poses_vggt(image_paths, device="cuda", conf_threshold=1.0)

    H, W = 800, 800
    K = torch.from_numpy(result.intrinsics[0].astype(np.float32)).cuda()

    # Scene geometry
    pts = result.point_cloud
    center = np.median(pts, axis=0)
    cam_positions = np.array([np.linalg.inv(result.extrinsics[i])[:3, 3] for i in range(len(result.extrinsics))])
    avg_dist = np.mean(np.linalg.norm(cam_positions - center, axis=1))

    # Estimate up from camera poses
    up_dirs = [np.linalg.inv(result.extrinsics[i])[:3, 1] for i in range(len(result.extrinsics))]
    avg_up = np.mean(up_dirs, axis=0)
    avg_up /= np.linalg.norm(avg_up)

    # Render training view re-renders for comparison
    print("Re-rendering training views...")
    gt_images = []
    re_renders = []
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
        re_renders.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    # Render 360 orbit
    print("Rendering 360 orbit...")
    novel_frames = []
    n_views = 90
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        # Orbit in the plane perpendicular to up
        arbitrary = np.array([1, 0, 0]) if abs(avg_up[0]) < 0.9 else np.array([0, 1, 0])
        basis1 = np.cross(avg_up, arbitrary)
        basis1 /= np.linalg.norm(basis1)
        basis2 = np.cross(avg_up, basis1)

        cam_pos = center + avg_dist * (np.cos(angle) * basis1 + np.sin(angle) * basis2)
        # Slight elevation variation
        cam_pos += avg_up * avg_dist * 0.15 * np.sin(angle * 0.3)

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
        novel_frames.append((renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))

    # Create GIFs
    gifs_dir = output_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training view comparison
    print("Creating comparison GIF...")
    combined = []
    for i in range(len(gt_images)):
        gt = cv2.resize(gt_images[i], (400, 400))
        re = cv2.resize(re_renders[i], (400, 400))
        lh = 30

        def label(img, text):
            out = np.zeros((400 + lh, 400, 3), dtype=np.uint8)
            out[:lh] = 25; out[lh:] = img
            cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
            return out

        sep = np.ones((400 + lh, 3, 3), dtype=np.uint8) * 50
        combined.append(np.concatenate([label(gt, "Ground Truth"), sep, label(re, "recon3d")], axis=1))

    pil = [Image.fromarray(f) for f in combined]
    comp_path = str(gifs_dir / "comparison.gif")
    pil[0].save(comp_path, save_all=True, append_images=pil[1:], duration=300, loop=0, optimize=True)
    print(f"  Comparison: {os.path.getsize(comp_path)/1024:.0f} KB")

    # 2. 360 orbit
    print("Creating orbit GIF...")
    pil_orbit = [Image.fromarray(cv2.resize(f, (512, 512))) for f in novel_frames]
    orbit_path = str(gifs_dir / "orbit.gif")
    pil_orbit[0].save(orbit_path, save_all=True, append_images=pil_orbit[1:], duration=50, loop=0, optimize=True)
    print(f"  Orbit: {os.path.getsize(orbit_path)/1024:.0f} KB")

    # 3. Combined: orbit with input thumbnail
    print("Creating hero GIF...")
    hero_frames = []
    for i in range(n_views):
        novel = cv2.resize(novel_frames[i], (512, 512))
        # Inset a training view thumbnail
        inp_idx = i % len(gt_images)
        thumb = cv2.resize(gt_images[inp_idx], (128, 128))
        # Add border
        bordered = np.zeros((132, 132, 3), dtype=np.uint8)
        bordered[:] = 200
        bordered[2:130, 2:130] = thumb
        novel[370:502, 10:142] = bordered
        cv2.putText(novel, "Input", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        hero_frames.append(novel)

    pil_hero = [Image.fromarray(f) for f in hero_frames]
    hero_path = str(gifs_dir / "hero.gif")
    pil_hero[0].save(hero_path, save_all=True, append_images=pil_hero[1:], duration=50, loop=0, optimize=True)
    print(f"  Hero: {os.path.getsize(hero_path)/1024:.0f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
