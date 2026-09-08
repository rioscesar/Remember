#!/usr/bin/env python3
"""Render a recovered wall from semantically selected, provenance-tracked source pixels."""

import argparse
import html
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import (
    View,
    angle_degrees,
    detect_planes,
    parse_camera,
    parse_views,
    project_world,
    quaternion_rotation,
    read_depth,
    slerp,
)

UNSUPPORTED = 0
CAPTURED_SINGLE_VIEW = 1
CAPTURED_MULTI_VIEW = 2
GEOMETRICALLY_INFERRED = 3
GENERATED = 4


def wall_mask(image: np.ndarray, processor, model, torch) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = processor.post_process_semantic_segmentation(
        type("Output", (), {"logits": logits})(),
        target_sizes=[image.shape[:2]],
    )[0].cpu().numpy()
    wall_ids = {
        int(identifier)
        for identifier, name in model.config.id2label.items()
        if name.lower().strip() in {"wall", "wall, brick"}
    }
    if not wall_ids:
        raise RuntimeError("The segmentation model does not expose an ADE20K wall class.")
    return np.isin(labels, list(wall_ids))


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    _, _, axes = np.linalg.svd(points - center, full_matrices=False)
    normal = axes[2]
    coordinates = (points - center) @ axes[:2].T
    return center, normal, axes[:2]


def geometry_labels(
    depth: np.ndarray,
    view: View,
    camera,
    plane_center: np.ndarray,
    plane_normal: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[y, x].astype(np.float64)
    camera_points = np.vstack((
        (x - camera.cx) / camera.fx * z,
        (y - camera.cy) / camera.fy * z,
        z,
    ))
    world_rays = view.rotation.T @ np.vstack((
        (x - camera.cx) / camera.fx,
        (y - camera.cy) / camera.fy,
        np.ones_like(z),
    ))
    world = (view.rotation.T @ camera_points + view.center[:, None]).T
    distances = np.abs((world - plane_center) @ plane_normal)
    valid_depth = np.zeros(depth.shape, dtype=bool)
    on_plane = np.zeros(depth.shape, dtype=bool)
    off_plane = np.zeros(depth.shape, dtype=bool)
    valid_depth[y, x] = True
    on_plane[y[distances <= threshold], x[distances <= threshold]] = True
    contradicted = distances >= threshold * 3
    off_plane[y[contradicted], x[contradicted]] = True
    return valid_depth, on_plane, off_plane


def target_wall_component(
    semantic: np.ndarray,
    plane_points: np.ndarray,
    view: View,
    camera,
) -> tuple[np.ndarray, int, np.ndarray]:
    projected, front = project_world(plane_points, view, camera)
    projected = np.rint(projected[front]).astype(np.int32)
    inside = (
        (projected[:, 0] >= 0) & (projected[:, 0] < camera.width)
        & (projected[:, 1] >= 0) & (projected[:, 1] < camera.height)
    )
    projected = projected[inside]
    seed = np.zeros(semantic.shape, dtype=np.uint8)
    seed[projected[:, 1], projected[:, 0]] = 255
    seed = cv2.dilate(seed, np.ones((5, 5), np.uint8))
    support_hull = np.zeros(semantic.shape, dtype=np.uint8)
    if len(projected) >= 3:
        hull = cv2.convexHull(projected.reshape(-1, 1, 2))
        cv2.fillConvexPoly(support_hull, hull, 255)
    component_count, labels = cv2.connectedComponents(semantic.astype(np.uint8), connectivity=8)
    selected = np.zeros(semantic.shape, dtype=bool)
    accepted_components = 0
    for label in range(1, component_count):
        component = labels == label
        if int((component & (seed > 0)).sum()) >= 10:
            selected |= component
            accepted_components += 1
    bounded = support_hull > 0
    return selected & bounded, accepted_components, bounded


def plane_homography(
    plane_points: np.ndarray,
    source: View,
    target: View,
    camera,
) -> tuple[np.ndarray, float, int]:
    source_pixels, source_front = project_world(plane_points, source, camera)
    target_pixels, target_front = project_world(plane_points, target, camera)
    valid = (
        source_front & target_front
        & (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < camera.width)
        & (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < camera.height)
        & (target_pixels[:, 0] >= 0) & (target_pixels[:, 0] < camera.width)
        & (target_pixels[:, 1] >= 0) & (target_pixels[:, 1] < camera.height)
    )
    source_pixels = source_pixels[valid].astype(np.float32)
    target_pixels = target_pixels[valid].astype(np.float32)
    if len(source_pixels) < 20:
        raise RuntimeError(f"{source.name} has insufficient plane observations.")
    homography, inliers = cv2.findHomography(source_pixels, target_pixels, cv2.RANSAC, 2.0)
    if homography is None:
        raise RuntimeError(f"Could not estimate wall homography for {source.name}.")
    accepted = inliers.ravel().astype(bool)
    if int(accepted.sum()) < 20:
        raise RuntimeError(f"{source.name} has too few planar homography inliers.")
    predicted = cv2.perspectiveTransform(source_pixels[accepted, None, :], homography)[:, 0]
    residual = np.linalg.norm(predicted - target_pixels[accepted], axis=1)
    rms = float(np.sqrt(np.mean(residual * residual)))
    if not math.isfinite(rms):
        raise RuntimeError(f"{source.name} has a non-finite planar homography residual.")
    return homography, rms, int(accepted.sum())


def target_plane_mask(
    plane_points: np.ndarray,
    plane_center: np.ndarray,
    plane_axes: np.ndarray,
    target: View,
    camera,
) -> np.ndarray:
    coordinates = (plane_points - plane_center) @ plane_axes.T
    low = np.percentile(coordinates, 2, axis=0)
    high = np.percentile(coordinates, 98, axis=0)
    corners_2d = np.array([
        [low[0], low[1]],
        [high[0], low[1]],
        [high[0], high[1]],
        [low[0], high[1]],
    ])
    corners_world = plane_center + corners_2d @ plane_axes
    pixels, front = project_world(corners_world, target, camera)
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    if front.all():
        cv2.fillConvexPoly(mask, np.rint(pixels).astype(np.int32), 255)
    result = mask > 0
    if not result.any():
        raise RuntimeError("The recovered wall has no visible footprint in the target camera.")
    return result


def blend_sources(layers: list[dict], plane_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = plane_mask.shape
    if not layers:
        raise RuntimeError("No source layer can observe the target wall.")
    colours = np.stack([layer["image"] for layer in layers]).astype(np.float32)
    valid = np.stack([layer["mask"] & plane_mask for layer in layers])
    weights = np.asarray([layer["weight"] for layer in layers], dtype=np.float32)[:, None, None]
    weighted_valid = np.where(valid, weights, 0)
    reference_index = np.argmax(weighted_valid, axis=0)
    reference = np.take_along_axis(
        colours,
        reference_index[None, :, :, None],
        axis=0,
    )[0]
    difference = np.mean(np.abs(colours - reference[None, :, :, :]), axis=3)
    consistent = valid & (difference <= 35)
    consistent_weight = np.where(consistent, weights, 0)
    support_count = consistent.sum(axis=0)
    weight_sum = consistent_weight.sum(axis=0)
    weighted = (colours * consistent_weight[:, :, :, None]).sum(axis=0)
    supported = support_count > 0
    colour = np.zeros((height, width, 3), dtype=np.uint8)
    colour[supported] = np.clip(weighted[supported] / weight_sum[supported, None], 0, 255).astype(np.uint8)

    raw_support_count = valid.sum(axis=0)
    disagreement = np.where(valid, difference, np.nan)
    overlap = np.broadcast_to(raw_support_count[None, :, :] >= 2, disagreement.shape)
    layer_indices = np.arange(len(layers))[:, None, None]
    is_reference = layer_indices == reference_index[None, :, :]
    disagreement_values = disagreement[overlap & ~is_reference & np.isfinite(disagreement)]
    high_conflict = (raw_support_count >= 2) & (support_count < raw_support_count)
    kernel = np.ones((5, 5), np.uint8)
    boundary = supported & ~(cv2.erode(supported.astype(np.uint8), kernel) > 0)
    plane_perimeter = cv2.morphologyEx(
        plane_mask.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        kernel,
    ) > 0
    boundary &= ~plane_perimeter

    provenance = np.full((height, width), UNSUPPORTED, dtype=np.uint8)
    provenance[support_count == 1] = CAPTURED_SINGLE_VIEW
    provenance[support_count >= 2] = CAPTURED_MULTI_VIEW
    total = height * width
    plane_pixels = max(1, int(plane_mask.sum()))
    metrics = {
        "directSourcePixelsPercent": float(supported.sum() / total * 100),
        "multiViewPercent": float((support_count >= 2).sum() / total * 100),
        "singleViewPercent": float((support_count == 1).sum() / total * 100),
        "unsupportedPercent": float((~supported).sum() / total * 100),
        "targetWallPixels": plane_pixels,
        "recoveredWallCoveragePercent": float((supported & plane_mask).sum() / plane_pixels * 100),
        "meanOverlapRgbDisagreement": float(np.mean(disagreement_values)) if disagreement_values.size else None,
        "highPhotometricConflictPercent": float(high_conflict.sum() / total * 100),
        "supportedPixelsWithConflictPercent": float(
            (high_conflict & supported).sum() / max(1, int(supported.sum())) * 100
        ),
        "foregroundBoundaryPixelsPercent": float(boundary.sum() / plane_pixels * 100),
        "boundaryConflictPercent": float(
            (boundary & high_conflict).sum() / max(1, int(boundary.sum())) * 100
        ),
        "generatedPixels": 0,
    }
    return colour, provenance, metrics


def write_report(output: Path, result: dict) -> None:
    cards = []
    for render in result["renders"]:
        index = render["view"]
        cards.append(f"""
        <section>
          <h2>Novel wall view {index}</h2>
          <div class="checker"><img src="wall-view-{index}.png"></div>
          <pre>{html.escape(json.dumps(render, indent=2))}</pre>
        </section>""")
    output.joinpath("index.html").write_text(f"""<!doctype html>
<meta charset="utf-8">
<title>Remember semantic wall spike</title>
<style>
body {{ margin: 0; background: #111; color: #eee; font: 16px system-ui; }}
header, section {{ padding: 20px; }}
.checker {{ max-width: 1200px; background: repeating-conic-gradient(#444 0 25%, #222 0 50%) 50% / 20px 20px; }}
img {{ display: block; width: 100%; }}
pre {{ white-space: pre-wrap; }}
</style>
<header>
  <h1>Semantic planar photographic reconstruction</h1>
  <p>Transparent pixels are unsupported. Visible pixels come directly from source photographs.</p>
  <p>Generated pixels: 0. Provenance maps: 0 unsupported, 1 captured single-view, 2 captured multi-view.</p>
</header>
{''.join(cards)}
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--depth-maps", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--mvs-order", type=Path, required=True)
    parser.add_argument("--model", default="openmmlab/upernet-convnext-tiny")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    args.output.mkdir(parents=True, exist_ok=True)
    camera = parse_camera(args.camera_text)
    views = parse_views(args.images_text)
    planes, models = detect_planes(args.ply, args.visibility, len(views))
    if not models:
        raise RuntimeError("No supported plane was recovered.")
    plane_points = models[0]["points"]
    plane_center, plane_normal, plane_axes = fit_plane(plane_points)
    plane_threshold = planes[0]["distanceThreshold"]
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    mvs_names = [
        line.strip()
        for line in args.mvs_order.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("__") and "," not in line
    ]
    mvs_index = {name: index for index, name in enumerate(mvs_names)}
    registered_names = {view.name for view in views}
    if len(mvs_names) != len(views) or set(mvs_names) != registered_names:
        raise RuntimeError("MVS image order does not match the registered camera views.")
    sources = []
    source_metrics = []
    for view in views:
        source_id = mvs_index.get(view.name)
        if source_id is None:
            continue
        visible_points = np.array([
            point
            for point, support in zip(plane_points, models[0]["supports"])
            if source_id in support
        ])
        if len(visible_points) < 100:
            continue
        image = cv2.imread(str(args.images / view.name), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read source image {view.name}.")
        depth = read_depth(args.depth_maps / f"{view.name}.geometric.bin")
        if depth.shape != (camera.height, camera.width):
            raise RuntimeError(f"Depth-map dimensions disagree with the camera for {view.name}.")
        semantic = wall_mask(image, processor, model, torch)
        valid_depth, on_plane, off_plane = geometry_labels(
            depth, view, camera, plane_center, plane_normal, plane_threshold
        )
        target_semantic, component_count, support_hull = target_wall_component(
            semantic, visible_points, view, camera
        )
        positive_total = max(1, int((on_plane & support_hull).sum()))
        semantic_total = max(1, int(target_semantic.sum()))
        semantic_with_depth = target_semantic & valid_depth
        decisive_geometry = semantic_with_depth & (on_plane | off_plane)
        decisive_total = max(1, int(decisive_geometry.sum()))
        metrics = {
            "name": view.name,
            "planeObservations": len(visible_points),
            "semanticWallPercent": float(semantic.mean() * 100),
            "targetWallComponents": component_count,
            "geometryWallRecallPercent": float((target_semantic & on_plane).sum() / positive_total * 100),
            "geometryConflictPercent": float((target_semantic & off_plane).sum() / semantic_total * 100),
            "depthValidatedPlaneAgreementPercent": float(
                (decisive_geometry & on_plane).sum() / decisive_total * 100
            ),
        }
        if (
            metrics["geometryWallRecallPercent"] < 20
            or metrics["geometryConflictPercent"] > 15
            or metrics["depthValidatedPlaneAgreementPercent"] < 55
        ):
            metrics["accepted"] = False
            source_metrics.append(metrics)
            continue
        # Depth-confirmed foreground always vetoes a semantic wall classification.
        geometry_veto = cv2.dilate(off_plane.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        refined = target_semantic & ~geometry_veto
        metrics["accepted"] = True
        source_metrics.append(metrics)
        cv2.imwrite(str(args.output / f"mask-{view.name}.png"), refined.astype(np.uint8) * 255)
        sources.append({"view": view, "image": image, "mask": refined, "points": visible_points})

    (args.output / "source-mask-metrics.json").write_text(
        json.dumps(source_metrics, indent=2),
        encoding="utf-8",
    )
    if len(sources) < 2:
        raise RuntimeError("Fewer than two views passed semantic/geometry wall-mask validation.")

    best_pair = min(
        (
            (first, second)
            for index, first in enumerate(sources)
            for second in sources[index + 1:]
        ),
        key=lambda pair: np.linalg.norm(pair[0]["view"].center - pair[1]["view"].center)
        * (1 + math.radians(math.degrees(math.acos(np.clip(
            np.dot(pair[0]["view"].forward, pair[1]["view"].forward), -1, 1
        ))))),
    )
    pair_distance = float(np.linalg.norm(best_pair[0]["view"].center - best_pair[1]["view"].center))
    pair_angle = angle_degrees(best_pair[0]["view"].forward, best_pair[1]["view"].forward)
    renders = []
    for render_index, fraction in enumerate((0.25, 0.5, 0.75), 1):
        first, second = best_pair[0]["view"], best_pair[1]["view"]
        target = View(
            f"virtual-{fraction}",
            quaternion_rotation(slerp(first.quaternion, second.quaternion, fraction)),
            first.center * (1 - fraction) + second.center * fraction,
            np.zeros(4),
            {},
        )
        plane_mask = target_plane_mask(plane_points, plane_center, plane_axes, target, camera)
        layers = []
        for source in sources:
            try:
                homography, rms, inliers = plane_homography(
                    source["points"], source["view"], target, camera
                )
            except RuntimeError:
                continue
            warped = cv2.warpPerspective(source["image"], homography, (camera.width, camera.height))
            warped_mask = cv2.warpPerspective(
                source["mask"].astype(np.float32),
                homography,
                (camera.width, camera.height),
                flags=cv2.INTER_LINEAR,
            ) >= 1.0 - 1e-6
            plane_ray = plane_center - source["view"].center
            plane_ray /= np.linalg.norm(plane_ray)
            view_angle = abs(float(np.dot(plane_ray, plane_normal)))
            weight = max(0.01, view_angle) * inliers / (1 + rms)
            layers.append({
                "image": warped,
                "mask": warped_mask,
                "weight": weight,
                "name": source["view"].name,
                "rms": rms,
                "inliers": inliers,
            })
        colour, provenance, metrics = blend_sources(layers, plane_mask)
        alpha = np.where(provenance > 0, 255, 0).astype(np.uint8)
        rgba = cv2.cvtColor(colour, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = alpha
        cv2.imwrite(str(args.output / f"wall-view-{render_index}.png"), rgba)
        cv2.imwrite(str(args.output / f"provenance-{render_index}.png"), provenance)
        cv2.imwrite(str(args.output / f"provenance-preview-{render_index}.png"), provenance * 90)
        metrics.update({
            "view": render_index,
            "fraction": fraction,
            "between": [first.name, second.name],
            "cameraDistance": pair_distance,
            "cameraAngularDistanceDegrees": pair_angle,
            "sourceLayers": [
                {"name": layer["name"], "rms": layer["rms"], "inliers": layer["inliers"]}
                for layer in layers
            ],
        })
        renders.append(metrics)

    result = {
        "model": args.model,
        "planePoints": len(plane_points),
        "planeThreshold": plane_threshold,
        "sourceMasks": source_metrics,
        "renders": renders,
        "provenance": {
            "0": "UNSUPPORTED",
            "1": "CAPTURED_SINGLE_VIEW",
            "2": "CAPTURED_MULTI_VIEW",
            "3": "GEOMETRICALLY_INFERRED",
            "4": "GENERATED",
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_report(args.output, result)
    print(f"Wrote semantic wall experiment to {args.output}")


if __name__ == "__main__":
    main()
