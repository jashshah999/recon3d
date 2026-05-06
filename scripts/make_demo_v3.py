"""Generate demo by rendering from the SAME camera poses VGGT estimated,
plus interpolated novel views between them."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path
from PIL import Image


def create_scene_images(output_dir: Path, n_images: int = 12):
    """Create textured multi-view images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    H, W = 512, 512
    fx = fy = 400.0
    cx, cy = W / 2, H / 2

    np.random.seed(42)

    # Dense point cloud with textures
    all_pts = []
    all_cols = []

    # Ground (grass-like)
    for _ in range(25000):
        x = np.random.uniform(-2.5, 2.5)
        z = np.random.uniform(-2.5, 2.5)
        y = 0.5 + np.random.randn() * 0.005
        g = 0.3 + 0.15 * (np.sin(x * 3) * np.cos(z * 4) + 1) / 2
        all_pts.append([x, y, z])
        all_cols.append([g * 0.5, g, g * 0.3])

    # Red sphere
    for _ in range(10000):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = 0.45
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) - 0.05
        z = r * np.cos(phi)
        n = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])
        light = np.array([0.5, -0.8, 0.3])
        light /= np.linalg.norm(light)
        d = max(0, np.dot(n, -light))
        br = 0.25 + 0.75 * d
        all_pts.append([x, y, z])
        all_cols.append([0.85 * br, 0.15 * br, 0.1 * br])

    # Blue box at (1.0, y, 0.5)
    for _ in range(8000):
        x = np.random.uniform(-0.3, 0.3) + 1.0
        y = np.random.uniform(-0.6, 0.5)
        z = np.random.uniform(-0.3, 0.3) + 0.5
        dx, dz = abs(x - 1.0), abs(z - 0.5)
        face_br = 0.7 if max(dx, dz) == dx else 0.5
        if abs(y - (-0.6)) < 0.02:
            face_br = 0.9
        all_pts.append([x, y, z])
        all_cols.append([0.08 * face_br, 0.12 * face_br, 0.75 * face_br])

    # Green cylinder at (-0.8, y, -0.5)
    for _ in range(6000):
        a = np.random.uniform(0, 2 * np.pi)
        r = 0.22
        x = r * np.cos(a) - 0.8
        z = r * np.sin(a) - 0.5
        y = np.random.uniform(-0.7, 0.5)
        n = np.array([np.cos(a), 0, np.sin(a)])
        light = np.array([0.5, -0.8, 0.3])
        light /= np.linalg.norm(light)
        d = max(0, np.dot(n, -light))
        br = 0.3 + 0.7 * d
        all_pts.append([x, y, z])
        all_cols.append([0.1 * br, 0.7 * br, 0.15 * br])

    # Brick wall at z=-2
    for _ in range(15000):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-1.5, 0.5)
        z = -2.0 + np.random.randn() * 0.005
        bx = int(x * 8) % 2
        by = int(y * 5) % 2
        if (bx + by) % 2 == 0:
            all_pts.append([x, y, z])
            all_cols.append([0.65, 0.35, 0.28])
        else:
            all_pts.append([x, y, z])
            all_cols.append([0.55, 0.28, 0.22])

    pts = np.array(all_pts, dtype=np.float32)
    cols = np.clip(np.array(all_cols, dtype=np.float32), 0, 1)

    image_paths = []
    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 3.5
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.5 + 0.3 * np.sin(angle * 0.5),
            radius * np.sin(angle),
        ])
        look_at = np.array([0, 0, 0])
        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        pts_cam = (R @ pts.T).T + t
        z_vals = pts_cam[:, 2]
        valid = z_vals > 0.1
        u = (fx * pts_cam[:, 0] / z_vals + cx)
        v = (fy * pts_cam[:, 1] / z_vals + cy)

        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Sky
        for row in range(H):
            f = row / H
            img[row, :] = [int(170 - 90 * f), int(195 - 70 * f), int(225 - 45 * f)]

        zbuf = np.full((H, W), 1e9, dtype=np.float32)
        order = np.argsort(-z_vals)
        for j in order:
            if not valid[j]:
                continue
            ui, vi = int(u[j]), int(v[j])
            if 0 <= ui < W and 0 <= vi < H and z_vals[j] < zbuf[vi, ui]:
                c = (cols[j] * 255).astype(np.uint8)
                s = max(1, int(5 / z_vals[j]))
                cv2.circle(img, (ui, vi), s, c.tolist(), -1)
                y1, y2 = max(0, vi - s), min(H, vi + s + 1)
                x1, x2 = max(0, ui - s), min(W, ui + s + 1)
                zbuf[y1:y2, x1:x2] = np.minimum(zbuf[y1:y2, x1:x2], z_vals[j])

        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_paths.append(str(path))

    return image_paths


def main():
    base_dir = Path("/tmp/recon3d_demo4")
    input_dir = base_dir / "input_frames"
    recon_dir = base_dir / "reconstruction"

    print("Creating scene...")
    image_paths = create_scene_images(input_dir, n_images=12)

    print("Reconstructing...")
    from recon3d.pipeline import reconstruct, PipelineConfig
    from recon3d.gaussian_train import TrainConfig

    config = PipelineConfig(
        pose_method="vggt",
        max_frames=12,
        metric_align=False,
        train_config=TrainConfig(max_steps=7000, log_every=1000),
        launch_viewer=False,
    )
    ply_path = reconstruct(str(input_dir), str(recon_dir), config)

    print("Rendering novel views...")
    from gsplat.rendering import rasterization

    ckpt = torch.load(str(recon_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    # Use the ACTUAL estimated poses to figure out the scene coordinate system
    from recon3d.pose_estimation import estimate_poses_vggt
    result = estimate_poses_vggt(image_paths, device="cuda", conf_threshold=1.0)

    # The scene center in VGGT's coordinate system
    pts = result.point_cloud
    scene_center = np.median(pts, axis=0)
    scene_radius = np.percentile(np.linalg.norm(pts - scene_center, axis=1), 90)

    print(f"Scene center: {scene_center}, radius: {scene_radius:.2f}")

    # Get the estimated camera positions to determine orbit radius
    cam_positions = []
    for i in range(len(result.extrinsics)):
        c2w = np.linalg.inv(result.extrinsics[i])
        cam_positions.append(c2w[:3, 3])
    cam_positions = np.array(cam_positions)
    avg_cam_dist = np.mean(np.linalg.norm(cam_positions - scene_center, axis=1))

    print(f"Avg camera distance from center: {avg_cam_dist:.2f}")

    H, W = 512, 512
    # Use estimated intrinsics from first frame
    K = torch.from_numpy(result.intrinsics[0].astype(np.float32)).cuda()

    render_frames = []
    input_renders = []  # Re-render from training viewpoints for comparison
    n_novel = 60

    # First render the training views for the side-by-side
    for i in range(len(result.extrinsics)):
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
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        input_renders.append(img)

    # Now render novel views orbiting around the scene
    up_vec = np.array([0, -1, 0], dtype=np.float32)  # VGGT uses y-down
    for i in range(n_novel):
        angle = 2 * np.pi * i / n_novel
        cam_pos = scene_center + avg_cam_dist * np.array([
            np.cos(angle),
            -0.2 + 0.15 * np.sin(angle * 0.7),
            np.sin(angle),
        ])

        forward = scene_center - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up_vec)
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
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh, viewmats=viewmat, Ks=K[None],
                width=W, height=H, sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        render_frames.append(img)

    # Create comparison GIF: GT input | re-rendered from same pose | novel view
    print("Creating GIFs...")
    combined = []
    input_paths = sorted(Path(input_dir).glob("frame_*.jpg"))
    for i in range(n_novel):
        inp_idx = i % len(input_paths)
        gt = cv2.cvtColor(cv2.imread(str(input_paths[inp_idx])), cv2.COLOR_BGR2RGB)
        rerender = input_renders[inp_idx % len(input_renders)]
        novel = render_frames[i]

        size = 256
        gt = cv2.resize(gt, (size, size))
        rerender = cv2.resize(rerender, (size, size))
        novel = cv2.resize(novel, (size, size))

        lh = 28
        def add_label(img, text):
            labeled = np.zeros((size + lh, size, 3), dtype=np.uint8)
            labeled[:lh] = 25
            labeled[lh:] = img
            cv2.putText(labeled, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            return labeled

        gt_l = add_label(gt, "Input")
        re_l = add_label(rerender, "Re-rendered")
        nv_l = add_label(novel, "Novel View")

        sep = np.ones((size + lh, 2, 3), dtype=np.uint8) * 50
        frame = np.concatenate([gt_l, sep, re_l, sep, nv_l], axis=1)
        combined.append(frame)

    pil = [Image.fromarray(f) for f in combined]
    gif3 = str(base_dir / "demo_3panel.gif")
    pil[0].save(gif3, save_all=True, append_images=pil[1:], duration=80, loop=0, optimize=True)
    print(f"3-panel: {gif3} ({os.path.getsize(gif3)/1024:.0f} KB)")

    # Novel view only
    pil_r = [Image.fromarray(cv2.resize(f, (512, 512))) for f in render_frames]
    rg = str(base_dir / "novel_views.gif")
    pil_r[0].save(rg, save_all=True, append_images=pil_r[1:], duration=66, loop=0, optimize=True)
    print(f"Novel views: {rg} ({os.path.getsize(rg)/1024:.0f} KB)")

    # Two panel: input | re-render (shows reconstruction quality)
    combined2 = []
    for i in range(len(input_paths)):
        gt = cv2.cvtColor(cv2.imread(str(input_paths[i])), cv2.COLOR_BGR2RGB)
        rerender = input_renders[i]
        size = 384
        gt = cv2.resize(gt, (size, size))
        rerender = cv2.resize(rerender, (size, size))
        lh = 32
        def add_label2(img, text):
            labeled = np.zeros((size + lh, size, 3), dtype=np.uint8)
            labeled[:lh] = 25; labeled[lh:] = img
            cv2.putText(labeled, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
            return labeled
        gt_l = add_label2(gt, "Ground Truth")
        re_l = add_label2(rerender, "recon3d Output")
        sep = np.ones((size + lh, 3, 3), dtype=np.uint8) * 50
        combined2.append(np.concatenate([gt_l, sep, re_l], axis=1))

    pil2 = [Image.fromarray(f) for f in combined2]
    gif2 = str(base_dir / "comparison.gif")
    pil2[0].save(gif2, save_all=True, append_images=pil2[1:], duration=500, loop=0, optimize=True)
    print(f"Comparison: {gif2} ({os.path.getsize(gif2)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
