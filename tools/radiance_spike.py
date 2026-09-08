#!/usr/bin/env python3
"""Milestone 0.4: local evidence-trained radiance reconstruction spike.

Trains a 3D Gaussian Splatting scene (gsplat, Apache-2.0) directly from the
existing registered COLMAP camera poses and sparse point cloud produced by
`reconstruct_memory.py`. No hosted training API, no generative image model,
and no monocular depth are used. All optimization is local, GPU-accelerated
CUDA rasterization against the person's own registered photographs.

Every rendered pixel in the output is RECONSTRUCTED: it is produced by a
radiance model fitted exclusively to captured multi-view photographs, not
generated content. The renderer separately reports per-pixel accumulated
alpha (opacity mass gathered along the ray) and each render's distance from
the nearest training camera, so unsupported/extrapolated regions can be
identified rather than presented as equally trustworthy evidence.

Source photographs, the trained checkpoint, and rendered private images
must remain outside the Git repository.
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import Camera, View, parse_camera, parse_views, angle_degrees


def quaternion_rotation_torch(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], dim=-1),
        torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], dim=-1),
        torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], dim=-1),
    ], dim=-2)


def slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    theta0 = math.acos(dot)
    theta = theta0 * fraction
    relative = second - first * dot
    relative /= np.linalg.norm(relative)
    return first * math.cos(theta) + relative * math.sin(theta)


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    positions, colors = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
        colors.append([float(fields[4]), float(fields[5]), float(fields[6])])
    return np.asarray(positions, dtype=np.float64), np.asarray(colors, dtype=np.float64) / 255.0


def nearest_neighbor_scale(points: np.ndarray) -> np.ndarray:
    # Small point counts: brute-force pairwise distances rather than adding a KD-tree dependency.
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


@dataclass
class TrainedScene:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor


def build_params(points: np.ndarray, colors: np.ndarray, device: str) -> dict:
    scale = np.clip(nearest_neighbor_scale(points), 1e-4, None)
    n = points.shape[0]
    params = {
        "means": torch.tensor(points, dtype=torch.float32, device=device),
        "scales": torch.log(torch.tensor(scale, dtype=torch.float32, device=device))[:, None].repeat(1, 3),
        "quats": torch.zeros((n, 4), dtype=torch.float32, device=device),
        "opacities": torch.logit(torch.full((n,), 0.1, dtype=torch.float32, device=device)),
        "colors": torch.logit(torch.tensor(colors, dtype=torch.float32, device=device).clamp(0.01, 0.99)),
    }
    params["quats"][:, 0] = 1.0
    return {name: torch.nn.Parameter(value) for name, value in params.items()}


def render(params: dict, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int):
    import gsplat
    colors, alphas, meta = gsplat.rasterization(
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
    )
    return colors[0].clamp(0, 1), alphas[0, ..., 0], meta


def view_to_matrices(view: View, camera: Camera, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = view.rotation
    viewmat[:3, 3] = view.rotation @ (-view.center)
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float32)
    return torch.tensor(viewmat, device=device), torch.tensor(K, device=device)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10 * math.log10(1.0 / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--points-text", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--holdout", type=str, default=None, help="Registered image name to hold out from training.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    camera = parse_camera(args.camera_text)
    views = [view for view in parse_views(args.images_text) if (args.images / view.name).exists()]
    if not views:
        raise RuntimeError("No registered views have a corresponding source image on disk.")

    scale = 1.0 / args.downsample
    render_camera = Camera(
        int(round(camera.width * scale)),
        int(round(camera.height * scale)),
        camera.fx * scale,
        camera.fy * scale,
        camera.cx * scale,
        camera.cy * scale,
    )

    holdout_view = None
    train_views = views
    if args.holdout:
        train_views = [view for view in views if view.name != args.holdout]
        holdout_matches = [view for view in views if view.name == args.holdout]
        holdout_view = holdout_matches[0] if holdout_matches else None
        if holdout_view is None:
            raise RuntimeError(f"Requested holdout view {args.holdout!r} is not a registered image.")

    images = {}
    for view in views:
        image = cv2.imread(str(args.images / view.name))
        image = cv2.resize(image, (render_camera.width, render_camera.height), interpolation=cv2.INTER_AREA)
        images[view.name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    points, colors = read_points3d(args.points_text)
    params = build_params(points, colors, device)

    import gsplat
    from gsplat.strategy import DefaultStrategy

    camera_centers = np.stack([view.center for view in train_views])
    scene_scale = float(np.linalg.norm(camera_centers.std(axis=0)))
    strategy = DefaultStrategy(verbose=False)
    strategy_state = strategy.initialize_state(scene_scale=max(scene_scale, 1e-3))

    lr_scale = max(scene_scale, 1e-3)
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4 * lr_scale, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2, eps=1e-15),
        "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3, eps=1e-15),
    }

    rng = np.random.default_rng(20260908)
    loss_log = []
    start = time.time()
    for step in range(args.iterations):
        view = train_views[rng.integers(0, len(train_views))]
        viewmat, K = view_to_matrices(view, render_camera, device)
        target = torch.tensor(images[view.name], device=device)

        rendered, alpha, meta = render(params, viewmat, K, render_camera.width, render_camera.height)
        loss = torch.abs(rendered - target).mean()

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        strategy.step_pre_backward(params, optimizers, strategy_state, step, meta)
        loss.backward()
        strategy.step_post_backward(params, optimizers, strategy_state, step, meta)
        for optimizer in optimizers.values():
            optimizer.step()

        if step % 200 == 0 or step == args.iterations - 1:
            loss_log.append({"step": step, "loss": float(loss.item()), "gaussians": int(params["means"].shape[0])})
            print(f"step {step:5d}  loss {loss.item():.4f}  gaussians {params['means'].shape[0]}", flush=True)

    training_seconds = time.time() - start

    torch.save({name: value.detach().cpu() for name, value in params.items()}, args.output / "checkpoint.pt")

    render_dir = args.output / "renders"
    render_dir.mkdir(exist_ok=True)

    def save_render(tag: str, viewmat: torch.Tensor, K: torch.Tensor, reference: np.ndarray | None, distance_to_nearest: float):
        with torch.no_grad():
            rendered, alpha, _ = render(params, viewmat, K, render_camera.width, render_camera.height)
        rgb = (rendered.cpu().numpy() * 255).astype(np.uint8)
        alpha_map = alpha.cpu().numpy()
        cv2.imwrite(str(render_dir / f"{tag}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(render_dir / f"{tag}-alpha.png"), (alpha_map * 255).astype(np.uint8))
        entry = {
            "tag": tag,
            "meanAccumulatedAlpha": float(alpha_map.mean()),
            "lowConfidencePixelPercent": float((alpha_map < 0.5).mean() * 100),
            "distanceToNearestTrainingCamera": distance_to_nearest,
        }
        if reference is not None:
            entry["psnrVsSourcePhoto"] = psnr(rendered.cpu().numpy(), reference)
        return entry

    render_reports = []

    # A: an original registered training camera pose.
    a_view = train_views[0]
    viewmat, K = view_to_matrices(a_view, render_camera, device)
    render_reports.append(save_render(
        "a-training-pose", viewmat, K, images[a_view.name],
        float(np.min(np.linalg.norm(camera_centers - a_view.center, axis=1))),
    ))

    # B: a held-out camera pose, if one was reserved.
    if holdout_view is not None:
        viewmat, K = view_to_matrices(holdout_view, render_camera, device)
        holdout_image = cv2.imread(str(args.images / holdout_view.name))
        holdout_image = cv2.resize(holdout_image, (render_camera.width, render_camera.height), interpolation=cv2.INTER_AREA)
        holdout_image = cv2.cvtColor(holdout_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        render_reports.append(save_render(
            "b-holdout-pose", viewmat, K, holdout_image,
            float(np.min(np.linalg.norm(camera_centers - holdout_view.center, axis=1))),
        ))

    # C: a small interpolated movement between two nearby evidence-supported cameras.
    pairs = [
        (first, second, np.linalg.norm(first.center - second.center))
        for index, first in enumerate(train_views) for second in train_views[index + 1:]
    ]
    first, second, _ = min(pairs, key=lambda item: item[2])
    for fraction in (0.25, 0.5, 0.75):
        center = first.center * (1 - fraction) + second.center * fraction
        quat = slerp(first.quaternion, second.quaternion, fraction)
        interpolated = View(f"interp-{fraction}", None, center, quat, {})
        rotation = quaternion_rotation_torch(torch.tensor(quat, dtype=torch.float32)).numpy()
        interpolated.rotation = rotation
        viewmat, K = view_to_matrices(interpolated, render_camera, device)
        render_reports.append(save_render(
            f"c-interpolated-{fraction:.2f}", viewmat, K, None,
            float(np.min(np.linalg.norm(camera_centers - center, axis=1))),
        ))

    # D: a view extrapolated beyond the supported camera coverage.
    centroid = camera_centers.mean(axis=0)
    farthest = train_views[0]
    for view in train_views:
        if np.linalg.norm(view.center - centroid) > np.linalg.norm(farthest.center - centroid):
            farthest = view
    direction = farthest.center - centroid
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    extrapolated_center = farthest.center + direction * scene_scale * 1.5
    extrapolated = View("d-extrapolated", farthest.rotation, extrapolated_center, farthest.quaternion, {})
    viewmat, K = view_to_matrices(extrapolated, render_camera, device)
    render_reports.append(save_render(
        "d-extrapolated", viewmat, K, None,
        float(np.min(np.linalg.norm(camera_centers - extrapolated_center, axis=1))),
    ))

    result = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "dataset": {
            "registeredViews": len(views),
            "trainingViews": len(train_views),
            "holdoutView": args.holdout,
            "initialSparsePoints": int(points.shape[0]),
            "renderWidth": render_camera.width,
            "renderHeight": render_camera.height,
        },
        "training": {
            "iterations": args.iterations,
            "trainingSeconds": training_seconds,
            "finalGaussianCount": int(params["means"].shape[0]),
            "lossLog": loss_log,
        },
        "renders": render_reports,
        "provenance": "Every rendered pixel is RECONSTRUCTED from Gaussians fitted exclusively to the registered source photographs. No pixel is GENERATED.",
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote radiance spike results to {args.output}")


if __name__ == "__main__":
    main()
