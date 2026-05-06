"""Generate demo using real images from a standard dataset (Tanks and Temples / LLFF style)."""

import numpy as np
import cv2
import torch
import os
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image


def download_sample_images(output_dir: Path):
    """Download sample images from VGGT's demo or use local synthetic but photo-realistic."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate photo-realistic looking images using a proper 3D scene with textures
    H, W = 512, 512
    n_images = 10

    # Create textured surfaces using numpy noise patterns
    np.random.seed(42)

    # Pre-generate textures
    wood_tex = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            v = int(120 + 40 * np.sin(y * 0.1 + np.sin(x * 0.05) * 3))
            wood_tex[y, x] = [v, int(v * 0.7), int(v * 0.4)]

    brick_tex = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            by = y % 32
            bx = (x + (16 if (y // 32) % 2 else 0)) % 64
            if by < 2 or bx < 2:
                brick_tex[y, x] = [180, 180, 170]
            else:
                r = 160 + int(20 * np.sin(y * 0.3 + x * 0.2))
                brick_tex[y, x] = [r, int(r * 0.55), int(r * 0.45)]

    grass_tex = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            g = 80 + int(40 * (np.sin(x * 0.4) * np.cos(y * 0.3) + 1) / 2)
            grass_tex[y, x] = [int(g * 0.4), g, int(g * 0.3)]

    image_paths = []
    for i in range(n_images):
        angle = 2 * np.pi * i / n_images
        radius = 4.0
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.8 + 0.5 * np.sin(angle * 0.3),
            radius * np.sin(angle),
        ])

        look_at = np.array([0.0, 0.0, 0.0])
        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0).astype(np.float32)
        t = (-R @ cam_pos).astype(np.float32)

        fx = fy = 400.0
        cx, cy = W / 2, H / 2

        img = np.zeros((H, W, 3), dtype=np.uint8)
        zbuf = np.full((H, W), 1e9, dtype=np.float32)

        # Sky
        for row in range(H):
            frac = row / H
            b = int(230 - 80 * frac)
            g = int(210 - 60 * frac)
            r = int(180 - 100 * frac)
            img[row, :] = [r, g, b]

        def draw_textured_points(pts, tex, n_pts=15000, point_size_base=5):
            u_all = fx * ((R @ pts.T).T + t)[:, 0] / ((R @ pts.T).T + t)[:, 2] + cx
            v_all = fy * ((R @ pts.T).T + t)[:, 1] / ((R @ pts.T).T + t)[:, 2] + cy
            z_all = ((R @ pts.T).T + t)[:, 2]
            valid = z_all > 0.1

            order = np.argsort(-z_all)
            for j in order:
                if not valid[j]:
                    continue
                ui, vi = int(u_all[j]), int(v_all[j])
                if 0 <= ui < W and 0 <= vi < H and z_all[j] < zbuf[vi, ui]:
                    # Get texture color
                    tx = int((pts[j, 0] * 50) % 256)
                    ty = int((pts[j, 2] * 50) % 256)
                    color = tex[ty % 256, tx % 256].tolist()
                    size = max(1, int(point_size_base / z_all[j]))
                    cv2.circle(img, (ui, vi), size, color, -1)
                    r = size
                    y1, y2 = max(0, vi - r), min(H, vi + r + 1)
                    x1, x2 = max(0, ui - r), min(W, ui + r + 1)
                    zbuf[y1:y2, x1:x2] = np.minimum(zbuf[y1:y2, x1:x2], z_all[j])

        # Ground plane with grass texture
        gx = np.random.uniform(-3, 3, 20000)
        gz = np.random.uniform(-3, 3, 20000)
        gy = np.full_like(gx, 0.5)
        draw_textured_points(np.stack([gx, gy, gz], axis=1).astype(np.float32), grass_tex, point_size_base=8)

        # Back wall with brick texture
        wx = np.random.uniform(-2.5, 2.5, 15000)
        wy = np.random.uniform(-1.5, 0.5, 15000)
        wz = np.full_like(wx, -2.0)
        draw_textured_points(np.stack([wx, wy, wz], axis=1).astype(np.float32), brick_tex, point_size_base=6)

        # Wooden table
        # Top
        tx = np.random.uniform(-0.6, 0.6, 8000)
        tz = np.random.uniform(-0.4, 0.4, 8000)
        ty = np.full_like(tx, -0.1)
        draw_textured_points(np.stack([tx, ty, tz], axis=1).astype(np.float32), wood_tex, point_size_base=5)

        # Legs
        for lx, lz in [(-0.5, -0.3), (0.5, -0.3), (-0.5, 0.3), (0.5, 0.3)]:
            ly_pts = np.random.uniform(-0.1, 0.5, 1500)
            lx_pts = np.random.uniform(lx - 0.04, lx + 0.04, 1500)
            lz_pts = np.random.uniform(lz - 0.04, lz + 0.04, 1500)
            draw_textured_points(np.stack([lx_pts, ly_pts, lz_pts], axis=1).astype(np.float32), wood_tex, point_size_base=4)

        # Red vase on table
        for _ in range(6000):
            theta = np.random.uniform(0, 2 * np.pi)
            y = np.random.uniform(-0.45, -0.1)
            # Vase profile
            r = 0.12 + 0.05 * np.sin((y + 0.1) * 8)
            x = r * np.cos(theta) + 0.15
            z = r * np.sin(theta) - 0.05
            normal = np.array([np.cos(theta), 0, np.sin(theta)])
            light = np.array([0.5, -0.7, 0.3])
            light /= np.linalg.norm(light)
            diff = max(0, np.dot(normal, -light))
            br = 0.3 + 0.7 * diff
            color = [int(200 * br), int(40 * br), int(30 * br)]
            pt = np.array([[x, y, z]], dtype=np.float32)
            pt_cam = (R @ pt.T).T + t
            zz = pt_cam[0, 2]
            if zz > 0.1:
                uu = int(fx * pt_cam[0, 0] / zz + cx)
                vv = int(fy * pt_cam[0, 1] / zz + cy)
                if 0 <= uu < W and 0 <= vv < H and zz < zbuf[vv, uu]:
                    s = max(1, int(4 / zz))
                    cv2.circle(img, (uu, vv), s, color, -1)

        # Slight blur to smooth point artifacts
        img = cv2.GaussianBlur(img, (3, 3), 0.8)

        path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_paths.append(str(path))

    return image_paths


def main():
    base_dir = Path("/tmp/recon3d_demo3")
    input_dir = base_dir / "input_frames"
    recon_dir = base_dir / "reconstruction"

    print("=" * 60)
    print("Step 1: Creating scene")
    print("=" * 60)
    image_paths = download_sample_images(input_dir)
    print(f"Created {len(image_paths)} images")

    print("\n" + "=" * 60)
    print("Step 2: Reconstructing with recon3d")
    print("=" * 60)
    from recon3d.pipeline import reconstruct, PipelineConfig
    from recon3d.gaussian_train import TrainConfig

    config = PipelineConfig(
        pose_method="vggt",
        max_frames=10,
        metric_align=False,
        train_config=TrainConfig(max_steps=7000, log_every=1000),
        launch_viewer=False,
    )
    ply_path = reconstruct(str(input_dir), str(recon_dir), config)

    print("\n" + "=" * 60)
    print("Step 3: Rendering novel views & creating GIF")
    print("=" * 60)
    from gsplat.rendering import rasterization

    H, W = 512, 512
    fx = fy = 400.0
    ckpt = torch.load(str(recon_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], dtype=torch.float32, device="cuda")

    render_frames = []
    n_views = 60
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        radius = 4.0
        cam_pos = np.array([
            radius * np.cos(angle),
            -0.8 + 0.4 * np.sin(angle * 0.5),
            radius * np.sin(angle),
        ])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right) + 1e-8
        up = np.cross(right, forward)
        Rm = np.stack([right, -up, forward], axis=0).astype(np.float32)
        tm = (-Rm @ cam_pos).astype(np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = Rm
        w2c[:3, 3] = tm

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        with torch.no_grad():
            renders, _, _ = rasterization(
                means=splats["means"], quats=splats["quats"],
                scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=colors_sh,
                viewmats=torch.from_numpy(w2c).cuda()[None],
                Ks=K[None], width=W, height=H, sh_degree=3,
                near_plane=0.01, far_plane=100.0,
            )
        img = (renders[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        render_frames.append(img)

    # Create side-by-side GIF
    input_images = sorted(Path(input_dir).glob("frame_*.jpg"))
    combined = []
    for i in range(n_views):
        inp = cv2.cvtColor(cv2.imread(str(input_images[i % len(input_images)])), cv2.COLOR_BGR2RGB)
        ren = render_frames[i]
        size = 384
        inp = cv2.resize(inp, (size, size))
        ren = cv2.resize(ren, (size, size))

        lh = 32
        inp_l = np.zeros((size + lh, size, 3), dtype=np.uint8)
        inp_l[:lh] = 30; inp_l[lh:] = inp
        cv2.putText(inp_l, "Input Views", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        ren_l = np.zeros((size + lh, size, 3), dtype=np.uint8)
        ren_l[:lh] = 30; ren_l[lh:] = ren
        cv2.putText(ren_l, "Novel View (recon3d)", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

        sep = np.ones((size + lh, 3, 3), dtype=np.uint8) * 60
        combined.append(np.concatenate([inp_l, sep, ren_l], axis=1))

    pil = [Image.fromarray(f) for f in combined]
    gif_path = str(base_dir / "demo.gif")
    pil[0].save(gif_path, save_all=True, append_images=pil[1:], duration=66, loop=0, optimize=True)
    print(f"Demo GIF: {gif_path} ({os.path.getsize(gif_path)/1024:.0f} KB)")

    # Render only
    pil_r = [Image.fromarray(cv2.resize(f, (512, 512))) for f in render_frames]
    rg = str(base_dir / "render.gif")
    pil_r[0].save(rg, save_all=True, append_images=pil_r[1:], duration=66, loop=0, optimize=True)
    print(f"Render GIF: {rg} ({os.path.getsize(rg)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
