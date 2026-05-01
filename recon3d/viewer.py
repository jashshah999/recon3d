"""Interactive 3D viewer using viser."""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


def launch_viewer(
    ply_path: str,
    image_paths: Optional[list[str]] = None,
    extrinsics: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
):
    """Launch an interactive web viewer for the reconstructed scene.

    Args:
        ply_path: Path to the .ply file with Gaussian splats.
        image_paths: Optional list of training image paths.
        extrinsics: Optional (N, 4, 4) world-to-camera matrices.
        intrinsics: Optional (N, 3, 3) intrinsic matrices.
        host: Host to bind the viewer to.
        port: Port to bind the viewer to.
    """
    try:
        import viser
    except ImportError:
        print("Install viser for the interactive viewer: pip install viser")
        return

    from plyfile import PlyData

    print(f"Loading {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)

    has_color = "red" in vertex or "f_dc_0" in vertex
    if "red" in vertex:
        colors = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]], axis=-1
        ).astype(np.float32) / 255.0
    elif "f_dc_0" in vertex:
        C0 = 0.28209479177387814
        colors = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=-1
        )
        colors = colors * C0 + 0.5
        colors = np.clip(colors, 0, 1)
    else:
        colors = np.ones_like(positions) * 0.5

    server = viser.ViserServer(host=host, port=port)
    print(f"\nViewer running at http://localhost:{port}")
    print("Press Ctrl+C to stop.\n")

    server.scene.add_point_cloud(
        "/points",
        points=positions[::max(1, len(positions) // 500_000)],
        colors=colors[::max(1, len(colors) // 500_000)],
        point_size=0.003,
    )

    if extrinsics is not None and image_paths is not None:
        for i in range(len(image_paths)):
            c2w = np.linalg.inv(extrinsics[i])
            position = c2w[:3, 3]
            rotation = c2w[:3, :3]

            wxyz = _rotation_matrix_to_quaternion(rotation)
            server.scene.add_frame(
                f"/cameras/cam_{i:04d}",
                wxyz=wxyz,
                position=position,
                axes_length=0.1,
                axes_radius=0.005,
            )

    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nViewer stopped.")


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to wxyz quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])
