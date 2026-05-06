"""Generate 360 demo by interpolating between actual VGGT-estimated camera poses."""

import numpy as np
import cv2
import torch
import os
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def interpolate_poses(poses_w2c, n_interp=90):
    """Smoothly interpolate between camera poses using slerp for rotations."""
    n_cams = len(poses_w2c)
    c2w_list = [np.linalg.inv(p) for p in poses_w2c]

    rotations = [Rotation.from_matrix(c[:3, :3]) for c in c2w_list]
    translations = [c[:3, 3] for c in c2w_list]

    # Close the loop
    rotations.append(rotations[0])
    translations.append(translations[0])

    interp_c2w = []
    for i in range(n_interp):
        t = (i / n_interp) * n_cams
        idx = int(t) % n_cams
        idx_next = (idx + 1) % (n_cams + 1)
        frac = t - int(t)

        # Slerp rotation
        key_rots = Rotation.concatenate([rotations[idx], rotations[idx_next]])
        slerp = Slerp([0, 1], key_rots)
        R_interp = slerp(frac).as_matrix()

        # Lerp translation
        t_interp = (1 - frac) * translations[idx] + frac * translations[idx_next]

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_interp
        c2w[:3, 3] = t_interp
        interp_c2w.append(c2w)

    return [np.linalg.inv(c).astype(np.float32) for c in interp_c2w]


def main():
    image_dir = "/tmp/lego_rgb"
    output_dir = Path("/tmp/recon3d_lego")

    from gsplat.rendering import rasterization
    from recon3d.pose_estimation import estimate_poses_vggt

    ckpt = torch.load(str(output_dir / "checkpoint.pt"), weights_only=False)
    splats = {k: torch.nn.Parameter(v.cuda()) for k, v in ckpt["splats"].items()}

    image_paths = sorted([str(p) for p in Path(image_dir).glob("*.png")])
    result = estimate_poses_vggt(image_paths, device="cuda", conf_threshold=1.0)

    H, W = 800, 800
    K = torch.from_numpy(result.intrinsics[0].astype(np.float32)).cuda()

    # Re-render training views
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

    # Interpolate between actual camera poses for smooth orbit
    print("Rendering interpolated novel views...")
    interp_w2c = interpolate_poses(result.extrinsics, n_interp=90)

    novel_frames = []
    for w2c in interp_w2c:
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

    gifs_dir = output_dir / "gifs_v2"
    gifs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Smooth orbit GIF
    print("Creating orbit GIF...")
    pil_orbit = [Image.fromarray(cv2.resize(f, (512, 512))) for f in novel_frames]
    orbit_path = str(gifs_dir / "orbit.gif")
    pil_orbit[0].save(orbit_path, save_all=True, append_images=pil_orbit[1:], duration=50, loop=0, optimize=True)
    print(f"  Orbit: {os.path.getsize(orbit_path)/1024:.0f} KB")

    # 2. Training view comparison
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

    # 3. Hero: orbit with inset training view
    print("Creating hero GIF...")
    hero_frames = []
    for i in range(len(novel_frames)):
        novel = cv2.resize(novel_frames[i], (512, 512))
        inp_idx = i % len(gt_images)
        thumb = cv2.resize(gt_images[inp_idx], (120, 120))
        bordered = np.zeros((124, 124, 3), dtype=np.uint8)
        bordered[:] = [180, 180, 180]
        bordered[2:122, 2:122] = thumb
        novel[378:502, 10:134] = bordered
        cv2.putText(novel, "Input", (18, 373), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        hero_frames.append(novel)
    pil_hero = [Image.fromarray(f) for f in hero_frames]
    hero_path = str(gifs_dir / "hero.gif")
    pil_hero[0].save(hero_path, save_all=True, append_images=pil_hero[1:], duration=50, loop=0, optimize=True)
    print(f"  Hero: {os.path.getsize(hero_path)/1024:.0f} KB")

    print("\nDone!")


if __name__ == "__main__":
    main()
