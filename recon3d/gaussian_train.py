"""Train 3D Gaussian Splatting from posed images and point cloud."""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    max_steps: int = 7000
    sh_degree: int = 3
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4
    ssim_weight: float = 0.2
    log_every: int = 500
    densify: bool = True
    max_gaussians: int = 250_000
    init_points: int = 150_000


def _knn(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    from sklearn.neighbors import NearestNeighbors
    x_np = x.cpu().numpy().astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x_np)
    distances, _ = nn.kneighbors(x_np)
    return torch.from_numpy(distances.astype(np.float32)).to(x.device)


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _init_splats(
    points: np.ndarray,
    colors: np.ndarray,
    sh_degree: int,
    device: str,
    max_points: int = 150_000,
) -> torch.nn.ParameterDict:
    """Initialize Gaussian splats from a point cloud."""
    pts = torch.from_numpy(points).float().to(device)
    rgbs = torch.from_numpy(colors).float().to(device)

    if len(pts) > max_points:
        indices = torch.randperm(len(pts))[:max_points]
        pts = pts[indices]
        rgbs = rgbs[indices]

    dist2_avg = (_knn(pts, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)

    n = pts.shape[0]
    quats = torch.rand((n, 4), device=device)
    opacities = torch.logit(torch.full((n,), 0.1, device=device))

    sh_coeffs = torch.zeros((n, (sh_degree + 1) ** 2, 3), device=device)
    sh_coeffs[:, 0, :] = _rgb_to_sh(rgbs)

    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh_coeffs[:, :1, :].contiguous()),
        "shN": torch.nn.Parameter(sh_coeffs[:, 1:, :].contiguous()),
    })


def train_gaussians(
    image_paths: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    point_cloud: np.ndarray,
    point_colors: np.ndarray,
    output_dir: str,
    config: Optional[TrainConfig] = None,
    device: str = "cuda",
) -> str:
    """Train 3D Gaussian Splatting model.

    Args:
        image_paths: Paths to training images.
        extrinsics: (N, 4, 4) world-to-camera matrices.
        intrinsics: (N, 3, 3) camera intrinsic matrices.
        point_cloud: (M, 3) initial point cloud.
        point_colors: (M, 3) point colors in [0, 1].
        output_dir: Directory to save outputs.
        config: Training configuration.
        device: Device to train on.

    Returns:
        Path to the exported .ply file.
    """
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy
    from gsplat import export_splats

    if config is None:
        config = TrainConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Initializing from {len(point_cloud)} points (capped at {config.init_points})...")
    splats = _init_splats(point_cloud, point_colors, config.sh_degree, device, config.init_points)

    scene_scale = float(np.percentile(np.linalg.norm(point_cloud, axis=1), 95))

    optimizers = {
        "means": torch.optim.Adam(
            [splats["means"]], lr=config.lr_means * scene_scale, eps=1e-15
        ),
        "scales": torch.optim.Adam([splats["scales"]], lr=config.lr_scales, eps=1e-15),
        "quats": torch.optim.Adam([splats["quats"]], lr=config.lr_quats, eps=1e-15),
        "opacities": torch.optim.Adam(
            [splats["opacities"]], lr=config.lr_opacities, eps=1e-15
        ),
        "sh0": torch.optim.Adam([splats["sh0"]], lr=config.lr_sh0, eps=1e-15),
        "shN": torch.optim.Adam([splats["shN"]], lr=config.lr_shN, eps=1e-15),
    }

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / config.max_steps)
    )

    strategy = DefaultStrategy(verbose=False)
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    images_gt = []
    viewmats = []
    Ks = []

    for i, path in enumerate(image_paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype(np.float32)
        images_gt.append(torch.from_numpy(img).float().to(device))

        w2c = extrinsics[i].astype(np.float32)
        viewmats.append(torch.from_numpy(w2c).to(device))
        Ks.append(torch.from_numpy(intrinsics[i].astype(np.float32)).to(device))

    n_images = len(image_paths)
    H, W = images_gt[0].shape[:2]

    print(f"Training {config.max_steps} steps on {n_images} images ({H}x{W})...")

    for step in range(config.max_steps):
        idx = torch.randint(0, n_images, (1,)).item()
        gt_image = images_gt[idx]
        viewmat = viewmats[idx][None]
        K = Ks[idx][None]

        img_h, img_w = gt_image.shape[:2]
        sh_degree_to_use = min(step // 1000, config.sh_degree)

        colors_sh = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors_sh,
            viewmats=viewmat,
            Ks=K,
            width=img_w,
            height=img_h,
            sh_degree=sh_degree_to_use,
            near_plane=0.01,
            far_plane=1e10,
            packed=False,
            absgrad=True,
        )

        rendered = renders[0]

        if config.densify:
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
            )

        l1_loss = F.l1_loss(rendered, gt_image)
        loss = (1 - config.ssim_weight) * l1_loss

        if config.ssim_weight > 0:
            try:
                from fused_ssim import fused_ssim
                ssim_val = fused_ssim(
                    rendered.permute(2, 0, 1)[None],
                    gt_image.permute(2, 0, 1)[None],
                )
                loss = loss + config.ssim_weight * (1 - ssim_val)
            except ImportError:
                pass

        loss.backward()

        if config.densify and len(splats["means"]) < config.max_gaussians:
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
                packed=False,
            )

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        lr_scheduler.step()

        if step % config.log_every == 0:
            n_gs = len(splats["means"])
            print(f"  Step {step}/{config.max_steps} | loss={loss.item():.4f} | gaussians={n_gs}")

    ply_path = str(output_dir / "scene.ply")
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),
        quats=splats["quats"].detach(),
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="ply",
        save_to=ply_path,
    )
    print(f"Exported: {ply_path}")

    splat_path = str(output_dir / "scene.splat")
    export_splats(
        means=splats["means"].detach(),
        scales=splats["scales"].detach(),
        quats=splats["quats"].detach(),
        opacities=splats["opacities"].detach(),
        sh0=splats["sh0"].detach(),
        shN=splats["shN"].detach(),
        format="splat",
        save_to=splat_path,
    )
    print(f"Exported: {splat_path}")

    checkpoint_path = str(output_dir / "checkpoint.pt")
    torch.save({
        "splats": {k: v.detach().cpu() for k, v in splats.items()},
        "config": config,
        "n_gaussians": len(splats["means"]),
    }, checkpoint_path)
    print(f"Checkpoint: {checkpoint_path}")

    del splats, optimizers
    torch.cuda.empty_cache()

    return ply_path
