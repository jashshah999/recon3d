"""Export reconstruction to COLMAP, nerfstudio, and other standard formats.

Lets users feed recon3d output directly into any 3DGS/NeRF pipeline.
"""

import os
import json
import struct
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def export_all(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str],
    points: np.ndarray,
    colors: np.ndarray,
    output_dir: str,
    image_sizes: list[tuple[int, int]] = None,
):
    """Export reconstruction to all supported formats."""
    export_colmap(poses_c2w, intrinsics, image_paths, points, colors, output_dir, image_sizes)
    export_nerfstudio(poses_c2w, intrinsics, image_paths, output_dir, image_sizes)
    export_ply(points, colors, os.path.join(output_dir, "scene.ply"))


def export_colmap(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str],
    points3d: np.ndarray,
    point_colors: np.ndarray,
    output_dir: str,
    image_sizes: list[tuple[int, int]] = None,
) -> str:
    """Export to COLMAP sparse text format."""
    sparse_dir = Path(output_dir) / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    N = len(poses_c2w)

    # cameras.txt
    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(N):
            K = intrinsics[i]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            if image_sizes:
                h, w = image_sizes[i]
            else:
                w, h = int(cx * 2), int(cy * 2)
            f.write(f"{i + 1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # images.txt
    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(N):
            w2c = np.linalg.inv(poses_c2w[i])
            quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            t = w2c[:3, 3]
            name = os.path.basename(image_paths[i])
            f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {i + 1} {name}\n")
            f.write("\n")

    # points3D.txt
    n_pts = min(len(points3d), 100000)
    indices = np.linspace(0, len(points3d) - 1, n_pts, dtype=int) if len(points3d) > n_pts else np.arange(len(points3d))
    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for idx, i in enumerate(indices):
            x, y, z = points3d[i]
            r, g, b = (np.clip(point_colors[i], 0, 1) * 255).astype(int)
            f.write(f"{idx + 1} {x} {y} {z} {r} {g} {b} 0.0\n")

    print(f"  COLMAP export: {sparse_dir}")
    return str(sparse_dir)


def export_nerfstudio(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    image_paths: list[str],
    output_dir: str,
    image_sizes: list[tuple[int, int]] = None,
) -> str:
    """Export to nerfstudio transforms.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    N = len(poses_c2w)
    K = intrinsics[0]

    if image_sizes:
        h, w = image_sizes[0]
    else:
        w, h = int(K[0, 2] * 2), int(K[1, 2] * 2)

    frames = []
    for i in range(N):
        c2w = poses_c2w[i].copy()
        c2w[:3, 1:3] *= -1  # OpenCV → OpenGL
        frames.append({
            "file_path": image_paths[i],
            "transform_matrix": c2w.tolist(),
        })

    transforms = {
        "fl_x": float(K[0, 0]),
        "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "w": w, "h": h,
        "aabb_scale": 16,
        "frames": frames,
    }

    out_path = output_dir / "transforms.json"
    with open(out_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"  nerfstudio export: {out_path}")
    return str(out_path)


def export_ply(points: np.ndarray, colors: np.ndarray, output_path: str) -> str:
    """Export colored point cloud to PLY."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    M = len(points)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {M}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(M):
            x, y, z = points[i]
            r, g, b = colors_uint8[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

    print(f"  PLY export: {output_path} ({M} points)")
    return str(output_path)
