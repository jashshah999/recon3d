"""Chunked VGGT pipeline with factor graph refinement.

VGGT can only process ~40 frames at once on 24GB VRAM. For longer videos,
this module chunks the sequence with overlap, runs VGGT per-chunk, then
stitches via Sim(3) alignment and refines with a GTSAM factor graph
for global consistency. Includes DINOv2 loop closure, multi-scale
confidence weighting, and robust outlier rejection.
"""

import numpy as np
import torch
import cv2
from typing import Optional

from .pose_estimation import PoseEstimationResult


def estimate_poses_chunked_vggt(
    image_paths: list[str],
    device: str = "cuda",
    conf_threshold: float = 1.5,
    chunk_size: int = 20,
    overlap: int = 5,
    use_factor_graph: bool = True,
    use_isam2: bool = True,
    loop_closure_threshold: float = 0.65,
    max_loop_closures: int = 50,
    robust_kernel: str = "cauchy",
) -> PoseEstimationResult:
    """Estimate poses using chunked VGGT with factor graph refinement.

    For sequences longer than chunk_size, splits into overlapping chunks,
    runs VGGT per-chunk, stitches via Sim(3), then refines globally with
    iSAM2 + DINOv2 loop closures.

    Args:
        image_paths: List of image file paths.
        device: CUDA device.
        conf_threshold: Confidence threshold for point filtering.
        chunk_size: Frames per VGGT forward pass.
        overlap: Overlap frames between consecutive chunks.
        use_factor_graph: Whether to apply GTSAM optimization.
        use_isam2: Use iSAM2 (incremental) instead of batch LM.
        loop_closure_threshold: DINOv2 similarity threshold for loop detection.
        max_loop_closures: Maximum number of loop closure constraints.
        robust_kernel: Robust loss for outlier rejection ("cauchy", "huber", "none").
    """
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    N = len(image_paths)
    step = chunk_size - overlap

    # Auto-adjust chunk size based on available VRAM
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if vram_gb < 10 and chunk_size > 8:
            chunk_size = 8
            step = chunk_size - overlap
            print(f"  Auto-reduced chunk_size to {chunk_size} for {vram_gb:.0f}GB GPU")
        elif vram_gb < 16 and chunk_size > 14:
            chunk_size = 14
            step = chunk_size - overlap
            print(f"  Auto-reduced chunk_size to {chunk_size} for {vram_gb:.0f}GB GPU")

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    chunks = []
    for start in range(0, N, step):
        end = min(start + chunk_size, N)
        if end - start < 3:
            break

        batch_paths = image_paths[start:end]
        print(f"  Chunk [{start}:{end}] ({end - start} frames)...")
        images = load_and_preprocess_images(batch_paths).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = model(images)

        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        ext_np = extrinsic.squeeze(0).cpu().numpy()
        int_np = intrinsic.squeeze(0).cpu().numpy()
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]

        # Build 4x4 w2c matrices
        n_chunk = ext_np.shape[0]
        w2c = np.zeros((n_chunk, 4, 4))
        w2c[:, :3, :] = ext_np
        w2c[:, 3, 3] = 1.0
        c2w = np.linalg.inv(w2c)

        # Unproject to world points
        world_pts = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )

        conf = depth_conf.squeeze(0).cpu().numpy()
        pose_conf = np.median(conf, axis=(1, 2))

        # Per-chunk scale normalization
        all_pts = world_pts.reshape(-1, 3).cpu().numpy()
        all_conf = conf.reshape(-1)
        valid = all_conf > 0.3
        valid_pts = all_pts[valid]
        valid_pts = valid_pts[np.isfinite(valid_pts).all(axis=-1)]
        mean_depth = float(np.mean(np.linalg.norm(valid_pts, axis=-1))) if len(valid_pts) > 0 else 1.0

        chunks.append({
            "start": start,
            "end": end,
            "c2w": c2w,
            "w2c": w2c,
            "intrinsics": int_np,
            "world_pts": world_pts.cpu().numpy(),
            "depth": depth[0].cpu().numpy(),
            "depth_conf": conf,
            "pose_conf": pose_conf,
            "mean_depth": mean_depth,
        })

        del predictions, images, depth, depth_conf
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    print(f"  {len(chunks)} chunks processed")

    # Stitch chunks via Sim(3) alignment on overlapping frames
    print("Stitching chunks via Sim(3) overlap alignment...")
    stitched_c2w = _naive_stitch(chunks, N)

    # Factor graph refinement
    if use_factor_graph:
        try:
            import gtsam  # noqa: F401
            print("Refining with factor graph...")
            if use_isam2:
                stitched_c2w = _isam2_refine(
                    chunks, N, stitched_c2w, image_paths,
                    loop_threshold=loop_closure_threshold,
                    max_loops=max_loop_closures,
                    robust_kernel=robust_kernel,
                )
            else:
                stitched_c2w = _factor_graph_refine(
                    chunks, N, stitched_c2w, image_paths,
                    loop_threshold=loop_closure_threshold,
                    max_loops=max_loop_closures,
                    robust_kernel=robust_kernel,
                )
        except ImportError:
            print("GTSAM not installed, skipping factor graph refinement.")
            print("Install with: pip install gtsam")

    # Convert c2w back to w2c extrinsics
    extrinsics = np.linalg.inv(stitched_c2w)

    # Merge intrinsics
    intrinsics = np.zeros((N, 3, 3))
    for chunk in chunks:
        for k in range(chunk["end"] - chunk["start"]):
            gi = chunk["start"] + k
            if np.allclose(intrinsics[gi], 0):
                intrinsics[gi] = chunk["intrinsics"][k]

    # Merge point clouds and depth maps
    all_points = []
    all_colors = []
    all_depths = [None] * N
    all_sizes = []

    for chunk in chunks:
        for k in range(chunk["end"] - chunk["start"]):
            gi = chunk["start"] + k
            if all_depths[gi] is not None:
                continue

            pts = chunk["world_pts"][k].reshape(-1, 3)
            c = chunk["depth_conf"][k].reshape(-1)

            img = cv2.cvtColor(cv2.imread(image_paths[gi]), cv2.COLOR_BGR2RGB)
            h_pts, w_pts = chunk["world_pts"][k].shape[:2]
            img_down = cv2.resize(img, (w_pts, h_pts))
            colors = (img_down / 255.0).reshape(-1, 3)

            valid = c > conf_threshold
            all_points.append(pts[valid])
            all_colors.append(colors[valid])

            all_depths[gi] = chunk["depth"][k, :, :, 0] if chunk["depth"][k].ndim == 3 else chunk["depth"][k]
            all_sizes.append((h_pts, w_pts))

    for i in range(N):
        if all_depths[i] is None:
            all_depths[i] = np.zeros((1, 1))
            all_sizes.append((1, 1))

    return PoseEstimationResult(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        point_cloud=np.concatenate(all_points, axis=0).astype(np.float32),
        point_colors=np.concatenate(all_colors, axis=0).astype(np.float32),
        depth_maps=all_depths,
        image_sizes=all_sizes,
        is_metric=False,
    )


def _naive_stitch(chunks: list, N: int) -> np.ndarray:
    """Stitch chunks by aligning overlapping frames via Sim(3)."""
    poses = np.zeros((N, 4, 4))
    poses[:] = np.eye(4)

    first = chunks[0]
    for i in range(first["end"] - first["start"]):
        poses[first["start"] + i] = first["c2w"][i]

    for c_idx in range(1, len(chunks)):
        chunk = chunks[c_idx]
        prev = chunks[c_idx - 1]

        overlap_start = chunk["start"]
        overlap_end = min(chunk["start"] + (prev["end"] - chunk["start"]), chunk["end"])
        n_overlap = overlap_end - overlap_start

        if n_overlap < 2:
            for i in range(chunk["end"] - chunk["start"]):
                gi = chunk["start"] + i
                if np.allclose(poses[gi], np.eye(4)):
                    poses[gi] = chunk["c2w"][i]
            continue

        prev_overlap = np.array([poses[overlap_start + k] for k in range(n_overlap)])
        curr_overlap = chunk["c2w"][:n_overlap]

        # Use confidence-weighted Procrustes
        conf_weights = chunk["pose_conf"][:n_overlap]
        T_align, scale = _procrustes_sim3(
            curr_overlap[:, :3, 3], prev_overlap[:, :3, 3], weights=conf_weights
        )
        R_align = T_align[:3, :3]
        t_align = T_align[:3, 3]

        for i in range(chunk["end"] - chunk["start"]):
            gi = chunk["start"] + i
            pose = chunk["c2w"][i].copy()
            aligned = np.eye(4)
            aligned[:3, :3] = R_align @ pose[:3, :3]
            aligned[:3, 3] = scale * R_align @ pose[:3, 3] + t_align

            if gi < overlap_end:
                w = (gi - overlap_start) / max(n_overlap, 1)
                poses[gi] = _interpolate_poses(poses[gi], aligned, w)
            else:
                poses[gi] = aligned

    return poses


def _isam2_refine(
    chunks: list, N: int, init_poses: np.ndarray, image_paths: list[str],
    loop_threshold: float = 0.65,
    max_loops: int = 50,
    robust_kernel: str = "cauchy",
) -> np.ndarray:
    """Refine stitched poses using iSAM2 (incremental Bayes tree)."""
    import gtsam

    params = gtsam.ISAM2Params()
    params.setRelinearizeThreshold(0.01)
    params.setRelinearizeSkip(1)
    isam = gtsam.ISAM2(params)

    # First pass: add odometry factors incrementally
    for i in range(N):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key_i = gtsam.symbol("x", i)

        pose_i = _mat_to_pose3(init_poses[i])
        values.insert(key_i, pose_i)

        if i == 0:
            graph.addPriorPose3(
                key_i, pose_i,
                gtsam.noiseModel.Isotropic.Sigma(6, 0.001),
            )
        else:
            # Odometry from chunk data
            rel, sigma = _get_odometry_from_chunks(chunks, i - 1, i, init_poses)
            noise = _make_noise(sigma, robust_kernel)
            graph.add(gtsam.BetweenFactorPose3(
                gtsam.symbol("x", i - 1), key_i,
                _mat_to_pose3(rel), noise
            ))

        isam.update(graph, values)

    # Second pass: add cross-chunk overlap constraints
    graph2 = gtsam.NonlinearFactorGraph()
    values2 = gtsam.Values()

    frame_chunks = _build_frame_to_chunk_map(chunks)
    for gi, chunk_list in frame_chunks.items():
        if len(chunk_list) < 2:
            continue
        for c_idx, k in chunk_list[1:]:
            chunk = chunks[c_idx]
            # Multi-view constraint: frame seen from two chunks
            pose_from_chunk = chunk["c2w"][k]
            rel = np.linalg.inv(init_poses[gi]) @ pose_from_chunk
            conf = float(chunk["pose_conf"][k])
            sigma = 0.05 / max(conf, 0.1)
            noise = _make_noise(sigma, robust_kernel)
            graph2.add(gtsam.BetweenFactorPose3(
                gtsam.symbol("x", gi), gtsam.symbol("x", gi),
                _mat_to_pose3(np.eye(4)), noise
            ))

    if graph2.size() > 0:
        isam.update(graph2, values2)

    # Third pass: loop closures
    n_loop_closures = _add_loop_closures_isam(
        isam, chunks, N, init_poses, image_paths,
        threshold=loop_threshold, max_loops=max_loops,
        robust_kernel=robust_kernel,
    )
    if n_loop_closures > 0:
        print(f"  Added {n_loop_closures} loop closures")

    # Extract optimized poses
    result = isam.calculateEstimate()
    optimized = np.zeros((N, 4, 4))
    for i in range(N):
        optimized[i] = _pose3_to_mat(result.atPose3(gtsam.symbol("x", i)))

    return optimized


def _factor_graph_refine(
    chunks: list, N: int, init_poses: np.ndarray, image_paths: list[str],
    loop_threshold: float = 0.65,
    max_loops: int = 50,
    robust_kernel: str = "cauchy",
) -> np.ndarray:
    """Refine stitched poses using a batch GTSAM factor graph (LM optimizer)."""
    import gtsam

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    key0 = gtsam.symbol("x", 0)
    graph.addPriorPose3(
        key0,
        _mat_to_pose3(init_poses[0]),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.001),
    )

    for i in range(N):
        values.insert(gtsam.symbol("x", i), _mat_to_pose3(init_poses[i]))

    # Within-chunk odometry factors (confidence-weighted)
    for chunk in chunks:
        for k in range(chunk["end"] - chunk["start"] - 1):
            i = chunk["start"] + k
            j = i + 1
            rel = np.linalg.inv(chunk["c2w"][k]) @ chunk["c2w"][k + 1]
            ki = gtsam.symbol("x", i)
            kj = gtsam.symbol("x", j)
            conf = min(chunk["pose_conf"][k], chunk["pose_conf"][k + 1])
            sigma = 0.02 / max(conf, 0.1)
            noise = _make_noise(sigma, robust_kernel)
            graph.add(gtsam.BetweenFactorPose3(ki, kj, _mat_to_pose3(rel), noise))

    # Cross-chunk overlap constraints
    frame_chunks = _build_frame_to_chunk_map(chunks)
    for gi, chunk_list in frame_chunks.items():
        if len(chunk_list) < 2:
            continue
        for c_idx, k in chunk_list[1:]:
            key_i = gtsam.symbol("x", gi)
            noise = _make_noise(0.1, robust_kernel)
            graph.addPriorPose3(key_i, _mat_to_pose3(init_poses[gi]), noise)

    # Loop closure via DINOv2
    n_loop_closures = _add_loop_closures_batch(
        graph, chunks, N, init_poses, image_paths,
        threshold=loop_threshold, max_loops=max_loops,
        robust_kernel=robust_kernel,
    )
    if n_loop_closures > 0:
        print(f"  Added {n_loop_closures} loop closures")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    params.setVerbosityLM("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    optimized = np.zeros((N, 4, 4))
    for i in range(N):
        optimized[i] = _pose3_to_mat(result.atPose3(gtsam.symbol("x", i)))

    return optimized


def _add_loop_closures_batch(
    graph, chunks: list, N: int, init_poses: np.ndarray, image_paths: list[str],
    threshold: float = 0.65, max_loops: int = 50, robust_kernel: str = "cauchy",
) -> int:
    """Add DINOv2 appearance-based loop closure factors (batch mode)."""
    import gtsam

    candidates = _detect_loop_candidates(image_paths, N, threshold, max_loops)
    if not candidates:
        return 0

    frame_to_chunk = _build_frame_to_chunk_index(chunks)
    n_added = 0

    for i, j, score in candidates:
        if not _verify_loop_closure_geometric(image_paths[i], image_paths[j]):
            continue

        ci, ki = frame_to_chunk.get(i, (None, None))
        cj, kj = frame_to_chunk.get(j, (None, None))

        if ci is not None and cj is not None and ci == cj:
            chunk = chunks[ci]
            rel = np.linalg.inv(chunk["c2w"][ki]) @ chunk["c2w"][kj]
            sigma = 0.1
        else:
            rel = np.linalg.inv(init_poses[i]) @ init_poses[j]
            sigma = 0.2

        key_i = gtsam.symbol("x", i)
        key_j = gtsam.symbol("x", j)
        noise = _make_noise(sigma, robust_kernel)
        graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _mat_to_pose3(rel), noise))
        n_added += 1

    return n_added


def _add_loop_closures_isam(
    isam, chunks: list, N: int, init_poses: np.ndarray, image_paths: list[str],
    threshold: float = 0.65, max_loops: int = 50, robust_kernel: str = "cauchy",
) -> int:
    """Add DINOv2 loop closure factors via iSAM2 update."""
    import gtsam

    candidates = _detect_loop_candidates(image_paths, N, threshold, max_loops)
    if not candidates:
        return 0

    frame_to_chunk = _build_frame_to_chunk_index(chunks)
    graph = gtsam.NonlinearFactorGraph()
    n_added = 0

    # Get current estimate for relative pose computation
    estimate = isam.calculateEstimate()

    for i, j, score in candidates:
        if not _verify_loop_closure_geometric(image_paths[i], image_paths[j]):
            continue

        ci, ki = frame_to_chunk.get(i, (None, None))
        cj, kj = frame_to_chunk.get(j, (None, None))

        if ci is not None and cj is not None and ci == cj:
            chunk = chunks[ci]
            rel = np.linalg.inv(chunk["c2w"][ki]) @ chunk["c2w"][kj]
            sigma = 0.08
        else:
            pose_i = _pose3_to_mat(estimate.atPose3(gtsam.symbol("x", i)))
            pose_j = _pose3_to_mat(estimate.atPose3(gtsam.symbol("x", j)))
            rel = np.linalg.inv(pose_i) @ pose_j
            sigma = 0.15

        # Scale sigma by inverse confidence (higher similarity = tighter constraint)
        sigma *= (1.0 - score + 0.3)

        key_i = gtsam.symbol("x", i)
        key_j = gtsam.symbol("x", j)
        noise = _make_noise(sigma, robust_kernel)
        graph.add(gtsam.BetweenFactorPose3(key_i, key_j, _mat_to_pose3(rel), noise))
        n_added += 1

    if n_added > 0:
        isam.update(graph, gtsam.Values())
        # Multiple optimization iterations for loop closure convergence
        isam.update()
        isam.update()

    return n_added


def _detect_loop_candidates(
    image_paths: list[str], N: int,
    threshold: float = 0.65, max_candidates: int = 50,
) -> list[tuple[int, int, float]]:
    """Detect loop closure candidates using DINOv2 appearance matching."""
    if N < 30:
        return []

    print("  Computing DINOv2 descriptors for loop closure...")
    descriptors = _compute_dinov2_descriptors(image_paths)
    sim = descriptors @ descriptors.T

    min_gap = max(15, N // 10)
    candidates = []
    for i in range(N):
        for j in range(i + min_gap, N):
            if sim[i, j] > threshold:
                candidates.append((i, j, float(sim[i, j])))

    candidates.sort(key=lambda x: -x[2])
    return candidates[:max_candidates]


def _compute_dinov2_descriptors(image_paths: list[str]) -> np.ndarray:
    """Extract L2-normalized DINOv2 CLS token descriptors."""
    import torch
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    descriptors = []
    batch_size = 16
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = []
        for path in batch_paths:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            imgs.append(img)

        tensors = torch.tensor(np.array(imgs), device=device, dtype=torch.float32)
        tensors = tensors.permute(0, 3, 1, 2)
        tensors = transform(tensors)

        with torch.no_grad():
            feats = model(tensors)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        descriptors.append(feats.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(descriptors, axis=0)


def _verify_loop_closure_geometric(img_path_i: str, img_path_j: str) -> bool:
    """Verify loop closure via ORB matching + fundamental matrix RANSAC."""
    img_i = cv2.imread(img_path_i, cv2.IMREAD_GRAYSCALE)
    img_j = cv2.imread(img_path_j, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create(1000)
    kp1, des1 = orb.detectAndCompute(img_i, None)
    kp2, des2 = orb.detectAndCompute(img_j, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if len(matches) < 15:
        return False

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)

    if mask is None:
        return False
    return int(mask.ravel().sum()) >= 20


def _build_frame_to_chunk_map(chunks: list) -> dict:
    """Map frame index → list of (chunk_idx, local_index) tuples."""
    frame_chunks = {}
    for c_idx, chunk in enumerate(chunks):
        for k in range(chunk["end"] - chunk["start"]):
            gi = chunk["start"] + k
            if gi not in frame_chunks:
                frame_chunks[gi] = []
            frame_chunks[gi].append((c_idx, k))
    return frame_chunks


def _build_frame_to_chunk_index(chunks: list) -> dict:
    """Map frame index → (first_chunk_idx, local_index)."""
    frame_to_chunk = {}
    for c_idx, chunk in enumerate(chunks):
        for k in range(chunk["end"] - chunk["start"]):
            gi = chunk["start"] + k
            if gi not in frame_to_chunk:
                frame_to_chunk[gi] = (c_idx, k)
    return frame_to_chunk


def _get_odometry_from_chunks(
    chunks: list, i: int, j: int, init_poses: np.ndarray
) -> tuple[np.ndarray, float]:
    """Get relative pose between frames i and j from chunk data."""
    for chunk in chunks:
        start, end = chunk["start"], chunk["end"]
        if start <= i < end and start <= j < end:
            ki = i - start
            kj = j - start
            rel = np.linalg.inv(chunk["c2w"][ki]) @ chunk["c2w"][kj]
            conf = min(chunk["pose_conf"][ki], chunk["pose_conf"][kj])
            sigma = 0.02 / max(conf, 0.1)
            return rel, sigma

    # Fallback: use stitched poses
    rel = np.linalg.inv(init_poses[i]) @ init_poses[j]
    return rel, 0.1


def _make_noise(sigma: float, robust_kernel: str = "cauchy"):
    """Create noise model with optional robust kernel."""
    import gtsam

    base_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
    if robust_kernel == "cauchy":
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), base_noise
        )
    elif robust_kernel == "huber":
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345), base_noise
        )
    return base_noise


def _procrustes_sim3(
    src: np.ndarray, dst: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple:
    """Sim(3) alignment via weighted Umeyama method."""
    if weights is None:
        weights = np.ones(len(src))
    weights = weights / weights.sum()

    src_c = np.average(src, axis=0, weights=weights)
    dst_c = np.average(dst, axis=0, weights=weights)
    src_centered = src - src_c
    dst_centered = dst - dst_c

    src_var = np.average(np.sum(src_centered ** 2, axis=1), weights=weights)
    if src_var < 1e-10:
        return np.eye(4), 1.0

    H = (src_centered * weights[:, None]).T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    scale = np.sum(S * np.diag(D)) / src_var
    t = dst_c - scale * R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, scale


def _interpolate_poses(p1: np.ndarray, p2: np.ndarray, w: float) -> np.ndarray:
    """SLERP interpolation of 4x4 poses."""
    from scipy.spatial.transform import Rotation, Slerp

    t = (1 - w) * p1[:3, 3] + w * p2[:3, 3]
    r1 = Rotation.from_matrix(p1[:3, :3])
    r2 = Rotation.from_matrix(p2[:3, :3])
    slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
    r = slerp(w)

    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def _mat_to_pose3(T):
    import gtsam
    return gtsam.Pose3(gtsam.Rot3(T[:3, :3]), gtsam.Point3(T[:3, 3]))


def _pose3_to_mat(pose):
    T = np.eye(4)
    T[:3, :3] = pose.rotation().matrix()
    T[:3, 3] = pose.translation()
    return T
