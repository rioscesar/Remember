#!/usr/bin/env python3
"""Milestone 0.5: sparse-view regularization + independent support gating.

Extends the Milestone 0.4 vanilla-gsplat spike (radiance_spike.py) with:

1. An evidence-compatible regularized training variant that uses only
   multi-view geometry already present in the COLMAP reconstruction
   (triangulated sparse-point depth, per-Gaussian scale/opacity priors) --
   no monocular depth network, no generative model, no new heavy
   dependency. This directly targets the Milestone 0.4 failure mode
   (needle-shaped floaters from too few training views).
2. A support/confidence gate that is INDEPENDENT of Gaussian opacity.
   For every rendered pixel, the expected ray depth is used to recover the
   3D world point actually being displayed, and that point's support is
   scored purely from camera geometry (distance and viewing-angle to the
   nearest training camera that could have observed it). Milestone 0.4
   showed splat alpha is not a trustworthy uncertainty signal, so this
   gate never looks at alpha/opacity when deciding SUPPORTED /
   WEAKLY_SUPPORTED / UNSUPPORTED.

Both the vanilla and regularized scenes are trained locally on the same
RTX 3070 laptop GPU used for Milestone 0.4. No Azure, no CUDA toolkit
compile step (gsplat prebuilt wheel only), no Android/AR changes.

Source photographs, checkpoints, and rendered private images must remain
outside the Git repository.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import Camera, View, parse_camera, parse_views
from radiance_spike import (
    build_params,
    nearest_neighbor_scale,
    psnr,
    quaternion_rotation_torch,
    read_points3d,
    slerp,
    view_to_matrices,
)


def read_points3d_by_id(path: Path) -> dict[int, np.ndarray]:
    """Parse points3D.txt keeping COLMAP point IDs for depth supervision."""
    points = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        points[int(fields[0])] = np.array([float(fields[1]), float(fields[2]), float(fields[3])])
    return points


def build_sparse_depth_targets(views: list[View], points_by_id: dict[int, np.ndarray], render_camera: Camera, source_camera: Camera) -> dict[str, np.ndarray]:
    """For each view, project its own observed 3D points back to pixel/depth pairs.

    This is pure multi-view geometry: the depth values come directly from
    the existing triangulated COLMAP points, not a learned monocular prior.
    """
    scale_x = render_camera.width / source_camera.width
    scale_y = render_camera.height / source_camera.height
    targets = {}
    for view in views:
        rows = []
        for point_id, (px, py) in view.observations.items():
            xyz = points_by_id.get(point_id)
            if xyz is None:
                continue
            camera_point = view.rotation @ (xyz - view.center)
            depth = float(camera_point[2])
            if depth <= 1e-4:
                continue
            u = px * scale_x
            v = py * scale_y
            if 0 <= u < render_camera.width and 0 <= v < render_camera.height:
                rows.append((u, v, depth))
        targets[view.name] = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)
    return targets


def render_rgbd(params: dict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int):
    import gsplat
    outputs, alphas, meta = gsplat.rasterization(
        params["means"],
        params["quats"] / params["quats"].norm(dim=-1, keepdim=True),
        torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]),
        torch.sigmoid(params["colors"]),
        viewmat[None],
        K[None],
        width,
        height,
        sh_degree=None,
        near_plane=0.001,
        rasterize_mode="antialiased",
        packed=False,
        render_mode="RGB+ED",
    )
    rgb = outputs[0, ..., :3].clamp(0, 1)
    depth = outputs[0, ..., 3]
    return rgb, depth, alphas[0, ..., 0], meta


def depth_supervision_loss(depth_map: torch.Tensor, targets: np.ndarray, device: str) -> torch.Tensor:
    if targets.shape[0] == 0:
        return torch.zeros((), device=device)
    us = torch.tensor(targets[:, 0], device=device).long().clamp(0, depth_map.shape[1] - 1)
    vs = torch.tensor(targets[:, 1], device=device).long().clamp(0, depth_map.shape[0] - 1)
    true_depth = torch.tensor(targets[:, 2], device=device)
    rendered_depth = depth_map[vs, us]
    valid = true_depth > 1e-4
    if valid.sum() == 0:
        return torch.zeros((), device=device)
    # Relative depth error: scene scale varies, absolute L1 would bias large-depth points.
    return (torch.abs(rendered_depth[valid] - true_depth[valid]) / true_depth[valid]).mean()


def scale_cap_loss(scales_log: torch.Tensor, cap: float) -> torch.Tensor:
    scales = torch.exp(scales_log)
    excess = torch.relu(scales - cap)
    return (excess ** 2).mean()


def opacity_entropy_loss(opacities_logit: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(opacities_logit).clamp(1e-4, 1 - 1e-4)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return entropy.mean()


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Small dependency-free SSIM using Gaussian-blurred local statistics (cv2 only)."""
    a = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return float(ssim_map.mean())


def train(views, train_views, images, points, colors, render_camera, device, iterations, regularized, points_by_id=None, source_camera=None, seed=20260908):
    import gsplat
    from gsplat.strategy import DefaultStrategy

    params = build_params(points, colors, device)
    camera_centers = np.stack([view.center for view in train_views])
    scene_scale = float(np.linalg.norm(camera_centers.std(axis=0)))
    lr_scale = max(scene_scale, 1e-3)

    if regularized:
        strategy = DefaultStrategy(
            verbose=False,
            prune_scale3d=0.05,          # tighter cap: prune long/needle-like Gaussians sooner
            refine_stop_iter=max(int(iterations * 0.4), 500),  # stop growing early with only ~10 views
            prune_opa=0.01,
        )
    else:
        strategy = DefaultStrategy(verbose=False)
    strategy_state = strategy.initialize_state(scene_scale=max(scene_scale, 1e-3))

    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * lr_scale, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2, eps=1e-15),
        "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3, eps=1e-15),
    }

    depth_targets = {}
    if regularized:
        depth_targets = build_sparse_depth_targets(train_views, points_by_id, render_camera, source_camera)

    scale_cap = 0.03 * lr_scale

    rng = np.random.default_rng(seed)
    loss_log = []
    start = time.time()
    for step in range(iterations):
        view = train_views[rng.integers(0, len(train_views))]
        viewmat, K = view_to_matrices(view, render_camera, device)
        target = torch.tensor(images[view.name], device=device)

        if regularized:
            rendered, depth_map, alpha, meta = render_rgbd(params, viewmat, K, render_camera.width, render_camera.height)
            loss = torch.abs(rendered - target).mean()
            loss = loss + 0.1 * depth_supervision_loss(depth_map, depth_targets.get(view.name, np.zeros((0, 3), dtype=np.float32)), device)
            loss = loss + 0.05 * scale_cap_loss(params["scales"], scale_cap)
            loss = loss + 0.01 * opacity_entropy_loss(params["opacities"])
        else:
            from radiance_spike import render as render_rgb
            rendered, alpha, meta = render_rgb(params, viewmat, K, render_camera.width, render_camera.height)
            loss = torch.abs(rendered - target).mean()

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        strategy.step_pre_backward(params, optimizers, strategy_state, step, meta)
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, meta)
        for optimizer in optimizers.values():
            optimizer.step()

        if step % 200 == 0 or step == iterations - 1:
            loss_log.append({"step": step, "loss": float(loss.item()), "gaussians": int(params["means"].shape[0])})
            print(f"[{'reg' if regularized else 'vanilla'}] step {step:5d}  loss {loss.item():.4f}  gaussians {params['means'].shape[0]}", flush=True)

    return params, loss_log, time.time() - start, scene_scale


def compute_world_points(depth_map: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject each pixel's rendered expected depth into world-space XYZ."""
    height, width = depth_map.shape
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (us - cx) / fx * depth_map
    y_cam = (vs - cy) / fy * depth_map
    z_cam = depth_map
    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
    rotation = viewmat[:3, :3]
    translation = viewmat[:3, 3]
    # X_cam = R X_world + t  =>  X_world = R^T (X_cam - t)
    world_points = (cam_points - translation) @ rotation
    return world_points


def support_gate(world_points: np.ndarray, query_center: np.ndarray, training_centers: np.ndarray, training_forwards: np.ndarray, max_distance: float, max_angle_deg: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Classify every pixel's displayed 3D point by camera-geometry evidence only.

    Never looks at rendered alpha/opacity. Returns (support_score in [0,1],
    label array of 0=UNSUPPORTED, 1=WEAKLY_SUPPORTED, 2=SUPPORTED).
    """
    height, width, _ = world_points.shape
    flat = world_points.reshape(-1, 3)
    best_score = np.zeros(flat.shape[0], dtype=np.float64)
    for center, forward in zip(training_centers, training_forwards):
        to_point = flat - center
        distance = np.linalg.norm(to_point, axis=1)
        direction = to_point / (distance[:, None] + 1e-9)
        cos_angle = np.clip((direction * forward).sum(axis=1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        distance_score = np.clip(1.0 - distance / max_distance, 0.0, 1.0)
        angle_score = np.clip(1.0 - angle_deg / max_angle_deg, 0.0, 1.0)
        score = distance_score * angle_score
        best_score = np.maximum(best_score, score)
    labels = np.zeros_like(best_score, dtype=np.int8)
    labels[best_score >= 0.55] = 2
    labels[(best_score >= 0.2) & (best_score < 0.55)] = 1
    return best_score.reshape(height, width), labels.reshape(height, width)


def fade_alpha(alpha: np.ndarray, support_score: np.ndarray) -> np.ndarray:
    """Attenuate alpha smoothly by support: full at >=0.55, ramps to 0 below 0.2."""
    fade = np.clip((support_score - 0.2) / (0.55 - 0.2), 0.0, 1.0)
    return alpha * fade


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--points-text", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--holdout", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    render_dir = args.output / "renders"
    render_dir.mkdir(exist_ok=True)

    source_camera = parse_camera(args.camera_text)
    views = [view for view in parse_views(args.images_text) if (args.images / view.name).exists()]
    if not views:
        raise RuntimeError("No registered views have a corresponding source image on disk.")

    scale = 1.0 / args.downsample
    render_camera = Camera(
        int(round(source_camera.width * scale)),
        int(round(source_camera.height * scale)),
        source_camera.fx * scale, source_camera.fy * scale,
        source_camera.cx * scale, source_camera.cy * scale,
    )

    train_views = views
    holdout_view = None
    if args.holdout:
        train_views = [view for view in views if view.name != args.holdout]
        matches = [view for view in views if view.name == args.holdout]
        holdout_view = matches[0] if matches else None
        if holdout_view is None:
            raise RuntimeError(f"Requested holdout view {args.holdout!r} is not registered.")

    images = {}
    for view in views:
        image = cv2.imread(str(args.images / view.name))
        image = cv2.resize(image, (render_camera.width, render_camera.height), interpolation=cv2.INTER_AREA)
        images[view.name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    points, colors = read_points3d(args.points_text)
    points_by_id = read_points3d_by_id(args.points_text)

    camera_centers = np.stack([view.center for view in train_views])
    camera_forwards = np.stack([view.forward for view in train_views])
    coverage_radius = float(np.linalg.norm(camera_centers - camera_centers.mean(axis=0), axis=1).max())
    max_distance = max(coverage_radius * 2.0, 1e-3)

    variants = {}
    for regularized in (False, True):
        params, loss_log, seconds, scene_scale = train(
            views, train_views, images, points, colors, render_camera, device,
            args.iterations, regularized, points_by_id=points_by_id, source_camera=source_camera,
        )
        variants[regularized] = {"params": params, "lossLog": loss_log, "trainingSeconds": seconds, "sceneScale": scene_scale}
        torch.save({name: value.detach().cpu() for name, value in params.items()},
                   args.output / f"checkpoint-{'regularized' if regularized else 'vanilla'}.pt")

    def render_variant(tag_prefix: str, params: dict, view: View, reference=None, gate=False):
        viewmat_np = np.eye(4, dtype=np.float32)
        viewmat_np[:3, :3] = view.rotation
        viewmat_np[:3, 3] = view.rotation @ (-view.center)
        K_np = np.array([[render_camera.fx, 0, render_camera.cx], [0, render_camera.fy, render_camera.cy], [0, 0, 1]], dtype=np.float32)
        viewmat = torch.tensor(viewmat_np, device=device)
        K = torch.tensor(K_np, device=device)
        with torch.no_grad():
            rgb, depth, alpha, _ = render_rgbd(params, viewmat, K, render_camera.width, render_camera.height)
        rgb_np = rgb.cpu().numpy()
        alpha_np = alpha.cpu().numpy()
        depth_np = depth.cpu().numpy()

        cv2.imwrite(str(render_dir / f"{tag_prefix}.png"), cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(render_dir / f"{tag_prefix}-alpha.png"), (alpha_np * 255).astype(np.uint8))

        entry = {
            "tag": tag_prefix,
            "meanAccumulatedAlpha": float(alpha_np.mean()),
            "lowConfidencePixelPercent": float((alpha_np < 0.5).mean() * 100),
            "distanceToNearestTrainingCamera": float(np.min(np.linalg.norm(camera_centers - view.center, axis=1))),
        }
        if reference is not None:
            entry["psnrVsSourcePhoto"] = psnr(rgb_np, reference)
            entry["ssimVsSourcePhoto"] = ssim(rgb_np, reference)

        if gate:
            world_points = compute_world_points(depth_np, viewmat_np, K_np)
            support_score, labels = support_gate(world_points, view.center, camera_centers, camera_forwards, max_distance)
            gated_alpha = fade_alpha(alpha_np, support_score)
            cv2.imwrite(str(render_dir / f"{tag_prefix}-support.png"), (support_score.clip(0, 1) * 255).astype(np.uint8))
            cv2.imwrite(str(render_dir / f"{tag_prefix}-gated-alpha.png"), (gated_alpha * 255).astype(np.uint8))
            gated_rgb = rgb_np * gated_alpha[..., None]
            cv2.imwrite(str(render_dir / f"{tag_prefix}-gated-rgb.png"), cv2.cvtColor((gated_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            total = labels.size
            entry["supportBreakdown"] = {
                "unsupportedPercent": float((labels == 0).mean() * 100),
                "weaklySupportedPercent": float((labels == 1).mean() * 100),
                "supportedPercent": float((labels == 2).mean() * 100),
            }
            entry["rawMeanAlphaInUnsupportedRegion"] = float(alpha_np[labels == 0].mean()) if (labels == 0).any() else None
            entry["gatedMeanAlphaInUnsupportedRegion"] = float(gated_alpha[labels == 0].mean()) if (labels == 0).any() else None
        return entry

    report = {"variants": {}}
    pairs = [
        (first, second, np.linalg.norm(first.center - second.center))
        for i, first in enumerate(train_views) for second in train_views[i + 1:]
    ]
    first_pair, second_pair, _ = min(pairs, key=lambda item: item[2])
    centroid = camera_centers.mean(axis=0)
    farthest = max(train_views, key=lambda v: np.linalg.norm(v.center - centroid))
    direction = (farthest.center - centroid)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    extrapolated_center = farthest.center + direction * variants[True]["sceneScale"] * 1.5
    extrapolated_view = View("extrapolated", farthest.rotation, extrapolated_center, farthest.quaternion, {})

    for regularized in (False, True):
        label = "regularized" if regularized else "vanilla"
        params = variants[regularized]["params"]
        renders = []
        renders.append(render_variant(f"{label}-a-training-pose", params, train_views[0], images[train_views[0].name]))
        if holdout_view is not None:
            holdout_image = cv2.imread(str(args.images / holdout_view.name))
            holdout_image = cv2.resize(holdout_image, (render_camera.width, render_camera.height), interpolation=cv2.INTER_AREA)
            holdout_image = cv2.cvtColor(holdout_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            renders.append(render_variant(f"{label}-b-holdout-pose", params, holdout_view, holdout_image, gate=True))
        quat = slerp(first_pair.quaternion, second_pair.quaternion, 0.5)
        center = first_pair.center * 0.5 + second_pair.center * 0.5
        interpolated = View("interp-0.50", quaternion_rotation_torch(torch.tensor(quat, dtype=torch.float32)).numpy(), center, quat, {})
        renders.append(render_variant(f"{label}-c-interpolated-0.50", params, interpolated))
        renders.append(render_variant(f"{label}-d-extrapolated", params, extrapolated_view, gate=True))
        report["variants"][label] = {
            "trainingSeconds": variants[regularized]["trainingSeconds"],
            "finalGaussianCount": int(params["means"].shape[0]),
            "lossLog": variants[regularized]["lossLog"],
            "renders": renders,
        }

    report["dataset"] = {
        "registeredViews": len(views),
        "trainingViews": len(train_views),
        "holdoutView": args.holdout,
        "initialSparsePoints": int(points.shape[0]),
        "renderWidth": render_camera.width,
        "renderHeight": render_camera.height,
    }
    report["supportGate"] = {
        "method": "per-pixel, from unprojected rendered depth vs. training-camera distance/viewing-angle only; independent of Gaussian opacity",
        "maxDistanceUsed": max_distance,
        "maxAngleDegUsed": 45.0,
        "thresholds": {"supported": ">=0.55", "weaklySupported": "0.2-0.55", "unsupported": "<0.2"},
    }
    report["provenance"] = "RECONSTRUCTED pixels only, fitted from registered photographs and their multi-view sparse-point depths. Support classification uses only camera geometry, never renderer opacity. No pixel is GENERATED or INFERRED from a monocular prior."
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Milestone 0.5 results to {args.output}")


if __name__ == "__main__":
    main()
