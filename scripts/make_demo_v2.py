"""Generate a better demo: render a 3D scene with OpenCV, reconstruct, render novel views."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path


def render_scene(cam_pos, look_at, H, W, fx, fy):
    """Render a synthetic scene with solid objects using z-buffer."""
    cx, cy = W / 2, H / 2

    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0, 1, 0])
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    R = np.stack([right, -up, forward], axis=0).astype(np.float32)
    t = (-R @ cam_pos).astype(np.float32)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    # Sky gradient
    for row in range(H):
        frac = row / H
        img[row, :] = [int(180 - 100 * frac), int(200 - 80 * frac), int(230 - 50 * frac)]

    def project(pts_world):
        pts_cam = (R @ pts_world.T).T + t
        z = pts_cam[:, 2]
        valid = z > 0.1
        u = (fx * pts_cam[:, 0] / z + cx)
        v = (fy * pts_cam[:, 1] / z + cy)
        return u, v, z, valid

    np.random.seed(42)

    # Ground plane - dense grid of colored points
    gx = np.linspace(-3, 3, 200)
    gz = np.linspace(-3, 3, 200)
    gx, gz = np.meshgrid(gx, gz)
    gx, gz = gx.ravel(), gz.ravel()
    gy = np.zeros_like(gx) + 0.5
    ground_pts = np.stack([gx, gy, gz], axis=1).astype(np.float32)
    # Checkerboard coloring
    ground_colors = np.zeros((len(gx), 3), dtype=np.uint8)
    for i in range(len(gx)):
        if (int(gx[i] * 2 + 100) % 2) == (int(gz[i] * 2 + 100) % 2):
            ground_colors[i] = [140, 160, 130]
        else:
            ground_colors[i] = [100, 120, 90]
    u, v, z, valid = project(ground_pts)
    order = np.argsort(-z)
    for j in order:
        if valid[j]:
            ui, vi = int(u[j]), int(v[j])
            if 0 <= ui < W and 0 <= vi < H and z[j] < zbuf[vi, ui]:
                size = max(2, int(8 / z[j]))
                cv2.circle(img, (ui, vi), size, ground_colors[j].tolist(), -1)
                zbuf[max(0, vi-size):min(H, vi+size), max(0, ui-size):min(W, ui+size)] = z[j]

    # Red sphere
    for _ in range(8000):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = 0.5
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) - 0.3
        zp = r * np.cos(phi)
        pt = np.array([[x, y, zp]], dtype=np.float32)
        # Lighting
        normal = np.array([x, y - (-0.3), zp]) / r
        light_dir = np.array([0.5, -0.8, 0.3])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = max(0, np.dot(normal, -light_dir))
        brightness = 0.3 + 0.7 * diffuse
        color = [int(220 * brightness), int(60 * brightness), int(50 * brightness)]

        pu, pv, pz, pvalid = project(pt)
        if pvalid[0]:
            ui, vi = int(pu[0]), int(pv[0])
            if 0 <= ui < W and 0 <= vi < H and pz[0] < zbuf[vi, ui]:
                size = max(1, int(4 / pz[0]))
                cv2.circle(img, (ui, vi), size, color, -1)
                zbuf[max(0, vi-size):min(H, vi+size), max(0, ui-size):min(W, ui+size)] = pz[0]

    # Blue box
    for _ in range(6000):
        x = np.random.uniform(-0.3, 0.3) + 1.0
        y = np.random.uniform(-0.3, 0.5)
        zp = np.random.uniform(-0.3, 0.3) + 0.5
        # Determine which face is visible
        dx = abs(x - 1.0)
        dy_top = abs(y - (-0.3))
        dz = abs(zp - 0.5)
        max_d = max(dx, dy_top, dz)
        if max_d == dx:
            brightness = 0.7 if x > 1.0 else 0.4
        elif max_d == dz:
            brightness = 0.6 if zp > 0.5 else 0.3
        else:
            brightness = 0.9
        color = [int(50 * brightness), int(80 * brightness), int(200 * brightness)]

        pt = np.array([[x, y, zp]], dtype=np.float32)
        pu, pv, pz, pvalid = project(pt)
        if pvalid[0]:
            ui, vi = int(pu[0]), int(pv[0])
            if 0 <= ui < W and 0 <= vi < H and pz[0] < zbuf[vi, ui]:
                size = max(1, int(3 / pz[0]))
                cv2.circle(img, (ui, vi), size, color, -1)
                zbuf[max(0, vi-size):min(H, vi+size), max(0, ui-size):min(W, ui+size)] = pz[0]

    # Green cylinder
    for _ in range(5000):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 0.25
        x = r * np.cos(angle) - 0.8
        zp = r * np.sin(angle) - 0.5
        y = np.random.uniform(-0.8, 0.5)
        normal = np.array([np.cos(angle), 0, np.sin(angle)])
        light_dir = np.array([0.5, -0.8, 0.3])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = max(0, np.dot(normal, -light_dir))
        brightness = 0.3 + 0.7 * diffuse
        color = [int(50 * brightness), int(180 * brightness), int(60 * brightness)]

        pt = np.array([[x, y, zp]], dtype=np.float32)
        pu, pv, pz, pvalid = project(pt)
        if pvalid[0]:
            ui, vi = int(pu[0]), int(pv[0])
            if 0 <= ui < W and 0 <= vi < H and pz[0] < zbuf[vi, ui]:
                size = max(1, int(3 / pz[0]))
                cv2.circle(img, (ui, vi), size, color, -1)
                zbuf[max(0, vi-size):min(H, vi+size), max(0, ui-size):min(W, ui+size)] = pz[0]

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R
    w2c[:3, 3] = t

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    return img, w2c, K


def main():
    base_dir = Path("/tmp/recon3d_demo2")
    input_dir = base_dir / "input_frames"
    recon_dir = base_dir / "reconstruction"
    render_dir = base_dir / "renders"
    input_dir.mkdir(parents=True, exist_ok=True)

    H, W = 512, 512
    fx = fy = 350.0
    n_images = 12
    look_at = np.array([0.0, 0.0, 0.0])

    print("=" * 60)
    print("Step 1: Creating synthetic multi-view scene")
    print("=" * 60)

    image_paths = []
    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 3.5
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.5 + 0.3 * np.sin(angle * 0.5),
            radius * np.sin(angle),
        ])

        img, w2c, K = render_scene(cam_pos, look_at, H, W, fx, fy)
        path = input_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_paths.append(str(path))

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
        train_config=TrainConfig(max_steps=5000, log_every=1000),
        launch_viewer=False,
        device="cuda",
    )
    ply_path = reconstruct(str(input_dir), str(recon_dir), config)

    print("\n" + "=" * 60)
    print("Step 3: Rendering novel views")
    print("=" * 60)
    from gsplat.rendering import rasterization

    render_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(str(recon_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    K_t = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32, device="cuda")

    render_frames = []
    n_views = 60
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        radius = 3.5
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.5 + 0.3 * np.sin(angle * 0.7),
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
                Ks=K_t[None],
                width=W, height=H,
                sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        render_frames.append(img)

    print(f"Rendered {len(render_frames)} novel views")

    print("\n" + "=" * 60)
    print("Step 4: Creating GIFs")
    print("=" * 60)
    from PIL import Image

    # Side by side: input cycling | novel view rendering
    input_images = sorted(Path(input_dir).glob("frame_*.jpg"))
    combined = []
    for i in range(n_views):
        inp_idx = i % len(input_images)
        inp = cv2.cvtColor(cv2.imread(str(input_images[inp_idx])), cv2.COLOR_BGR2RGB)
        ren = render_frames[i]

        # Resize to 384
        size = 384
        inp = cv2.resize(inp, (size, size))
        ren = cv2.resize(ren, (size, size))

        # Labels
        lh = 32
        inp_l = np.zeros((size + lh, size, 3), dtype=np.uint8)
        inp_l[:lh] = 30
        inp_l[lh:] = inp
        cv2.putText(inp_l, "Input Views", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        ren_l = np.zeros((size + lh, size, 3), dtype=np.uint8)
        ren_l[:lh] = 30
        ren_l[lh:] = ren
        cv2.putText(ren_l, "Novel View (recon3d)", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        sep = np.ones((size + lh, 3, 3), dtype=np.uint8) * 60
        frame = np.concatenate([inp_l, sep, ren_l], axis=1)
        combined.append(frame)

    pil = [Image.fromarray(f) for f in combined]
    gif_path = str(base_dir / "demo.gif")
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=66, loop=0, optimize=True)
    print(f"Side-by-side: {gif_path} ({os.path.getsize(gif_path) / 1024:.0f} KB)")

    # Render only
    pil_r = [Image.fromarray(cv2.resize(f, (512, 512))) for f in render_frames]
    rg = str(base_dir / "render.gif")
    pil_r[0].save(rg, save_all=True, append_images=pil_r[1:], duration=66, loop=0, optimize=True)
    print(f"Render only: {rg} ({os.path.getsize(rg) / 1024:.0f} KB)")

    print("\nDone!")


if __name__ == "__main__":
    main()
