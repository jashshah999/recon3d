"""Align VGGT's relative-scale output to metric scale using MoGe-2 or MoGe-3."""

import importlib

import cv2
import numpy as np
import torch

# Short names accepted by --metric-model -> (moge.model module, Hugging Face repo)
METRIC_MODELS = {
    "moge-2": ("v2", "Ruicheng/moge-2-vitl"),
    "moge-3": ("v3", "Ruicheng/moge-3-vitl"),
    "moge-3-vitg": ("v3", "Ruicheng/moge-3-vitg"),
}


def resolve_metric_model(name: str) -> tuple[str, str]:
    """Map a short name, Hugging Face repo or checkpoint path to (moge version, pretrained)."""
    if name in METRIC_MODELS:
        return METRIC_MODELS[name]
    lowered = name.lower()
    if "moge-3" in lowered or "moge3" in lowered:
        return "v3", name
    if "moge-2" in lowered or "moge2" in lowered:
        return "v2", name
    raise ValueError(
        f"Can't tell which MoGe version '{name}' is. Use one of {sorted(METRIC_MODELS)} "
        "or a repo/path containing 'moge-2' or 'moge-3' (MoGe-1 has no metric scale)."
    )


def load_metric_model(name: str, device: str):
    version, pretrained = resolve_metric_model(name)
    try:
        module = importlib.import_module(f"moge.model.{version}")
    except (ImportError, RuntimeError) as e:
        # FlexGEMM (MoGe-3) raises RuntimeError at import time when no CUDA GPU is visible.
        hint = " MoGe-3 needs MoGe >= 3.0 and a CUDA GPU (FlexGEMM/Triton)." if version == "v3" else ""
        raise ImportError(
            f"Could not import moge.model.{version}: {e}.{hint} "
            "Install/upgrade with: pip install -U git+https://github.com/microsoft/MoGe.git"
        ) from e
    return version, module.MoGeModel.from_pretrained(pretrained).to(device).eval()


def align_to_metric(
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    point_cloud: np.ndarray,
    depth_maps: list[np.ndarray],
    device: str = "cuda",
    n_reference_frames: int = 5,
    metric_model: str = "moge-2",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Align relative-scale reconstruction to metric scale using MoGe.

    Uses MoGe metric depth on a subset of frames, then computes a global
    scale factor to convert the entire reconstruction to metric.

    Args:
        image_paths: Paths to input images.
        extrinsics: (N, 4, 4) world-to-camera matrices.
        intrinsics: (N, 3, 3) camera intrinsic matrices.
        point_cloud: (M, 3) point cloud in relative scale.
        depth_maps: List of (H, W) depth maps in relative scale.
        device: Device to run MoGe on.
        n_reference_frames: Number of frames to use for scale estimation.
        metric_model: "moge-2", "moge-3", "moge-3-vitg", or a MoGe repo/checkpoint path.

    Returns:
        Tuple of (extrinsics, intrinsics, point_cloud, depth_maps) in metric scale.
    """
    print(f"Loading {metric_model} for metric alignment...")
    version, model = load_metric_model(metric_model, device)

    on_cuda = str(device).startswith("cuda")
    if version == "v2":
        # MoGe-2 defaults to fp16, which CPU convolutions don't support.
        infer_kwargs = {"use_fp16": on_cuda}
    else:
        infer_kwargs = {}
        if not on_cuda:
            # MoGe-3's sparse refinement runs through FlexGEMM/Triton kernels (CUDA only).
            print("  MoGe-3 on CPU: skipping sparse refinement (needs CUDA)")
            infer_kwargs["refine_steps"] = 0

    n_frames = len(image_paths)
    ref_indices = np.linspace(0, n_frames - 1, min(n_reference_frames, n_frames), dtype=int)

    scale_ratios = []

    for idx in ref_indices:
        image = cv2.cvtColor(cv2.imread(image_paths[idx]), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(
            image / 255.0, dtype=torch.float32, device=device
        ).permute(2, 0, 1)

        with torch.no_grad():
            output = model.infer(image_tensor, **infer_kwargs)

        metric_depth = output["depth"].float().cpu().numpy()

        rel_depth = depth_maps[idx]

        if metric_depth.shape != rel_depth.shape:
            metric_depth = cv2.resize(
                metric_depth, (rel_depth.shape[1], rel_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        valid = (rel_depth > 1e-6) & (metric_depth > 1e-6)
        if valid.sum() < 100:
            continue

        ratio = np.median(metric_depth[valid] / rel_depth[valid])
        if 0.01 < ratio < 1000:
            scale_ratios.append(ratio)

    del model
    torch.cuda.empty_cache()

    if not scale_ratios:
        print("WARNING: Could not estimate metric scale. Using relative scale.")
        return extrinsics, intrinsics, point_cloud, depth_maps

    scale = np.median(scale_ratios)
    print(f"Metric scale factor: {scale:.4f}")

    scaled_extrinsics = extrinsics.copy()
    scaled_extrinsics[:, :3, 3] *= scale

    scaled_points = point_cloud * scale
    scaled_depths = [d * scale for d in depth_maps]

    return scaled_extrinsics, intrinsics, scaled_points, scaled_depths
