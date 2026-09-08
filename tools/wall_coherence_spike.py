#!/usr/bin/env python3
"""Build a stable, source-derived photographic atlas for one recovered wall."""

import argparse
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
from semantic_plane_spike import (
    CAPTURED_MULTI_VIEW,
    CAPTURED_SINGLE_VIEW,
    UNSUPPORTED,
    fit_plane,
    geometry_labels,
    target_wall_component,
    wall_mask,
)


def homography(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, int]:
    matrix, inliers = cv2.findHomography(
        source.astype(np.float32),
        target.astype(np.float32),
        cv2.RANSAC,
        2.0,
    )
    if matrix is None:
        raise RuntimeError("Could not estimate planar homography.")
    accepted = inliers.ravel().astype(bool)
    if int(accepted.sum()) < 20:
        raise RuntimeError("Too few planar homography inliers.")
    predicted = cv2.perspectiveTransform(source[accepted, None].astype(np.float32), matrix)[:, 0]
    residual = np.linalg.norm(predicted - target[accepted], axis=1)
    rms = float(np.sqrt(np.mean(residual * residual)))
    if not math.isfinite(rms):
        raise RuntimeError("Non-finite planar homography residual.")
    return matrix, rms, int(accepted.sum())


def atlas_coordinates(
    points: np.ndarray,
    center: np.ndarray,
    axes: np.ndarray,
    width: int,
) -> tuple[np.ndarray, dict]:
    coordinates = (points - center) @ axes.T
    low = np.percentile(coordinates, 2, axis=0)
    high = np.percentile(coordinates, 98, axis=0)
    span = high - low
    height = max(1, round(width * span[1] / span[0]))
    pixels = np.column_stack((
        (coordinates[:, 0] - low[0]) / span[0] * (width - 1),
        (high[1] - coordinates[:, 1]) / span[1] * (height - 1),
    ))
    return pixels, {
        "width": width,
        "height": height,
        "low": low.tolist(),
        "high": high.tolist(),
    }


def robust_color_transform(source: np.ndarray, reference: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample = np.flatnonzero(overlap.ravel())
    if sample.size < 500:
        return np.ones(3), np.zeros(3)
    if sample.size > 100_000:
        sample = sample[np.linspace(0, sample.size - 1, 100_000).astype(int)]
    source_values = source.reshape(-1, 3)[sample].astype(np.float64)
    reference_values = reference.reshape(-1, 3)[sample].astype(np.float64)
    source_median = np.median(source_values, axis=0)
    reference_median = np.median(reference_values, axis=0)
    gain = np.divide(
        reference_median,
        source_median,
        out=np.ones(3),
        where=source_median >= 8,
    )
    # A small per-channel gain compensates exposure and white balance while preserving
    # captured contrast; chained gain+bias fitting visibly washed out this wall.
    return np.clip(gain, 0.9, 1.1), np.zeros(3)


def apply_color_transform(image: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def mean_pairwise_difference(images: list[np.ndarray], masks: list[np.ndarray]) -> float:
    differences = []
    for index, first in enumerate(images):
        for second_index in range(index + 1, len(images)):
            overlap = masks[index] & masks[second_index]
            if overlap.sum() < 100:
                continue
            difference = np.mean(
                np.abs(first[overlap].astype(np.float32) - images[second_index][overlap].astype(np.float32)),
                axis=1,
            )
            differences.append(difference)
    return float(np.mean(np.concatenate(differences))) if differences else 0.0


def seam_magnitude(image: np.ndarray, ownership: np.ndarray) -> tuple[float, np.ndarray]:
    seam = np.zeros(ownership.shape, dtype=bool)
    seam[:, 1:] |= (ownership[:, 1:] != ownership[:, :-1]) & (ownership[:, 1:] > 0) & (ownership[:, :-1] > 0)
    seam[1:, :] |= (ownership[1:, :] != ownership[:-1, :]) & (ownership[1:, :] > 0) & (ownership[:-1, :] > 0)
    gradient_x = np.mean(np.abs(image[:, 1:].astype(np.float32) - image[:, :-1].astype(np.float32)), axis=2)
    gradient_y = np.mean(np.abs(image[1:].astype(np.float32) - image[:-1].astype(np.float32)), axis=2)
    values = []
    values.extend(gradient_x[seam[:, 1:]].tolist())
    values.extend(gradient_y[seam[1:, :]].tolist())
    return (float(np.mean(values)) if values else 0.0), seam


def build_atlas(
    layers: list[dict],
    atlas_shape: tuple[int, int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    height, width = atlas_shape
    order = sorted(range(len(layers)), key=lambda index: layers[index]["baseScore"], reverse=True)
    reference = layers[order[0]]["image"].copy()
    reference_mask = layers[order[0]]["mask"].copy()
    for index in order:
        layer = layers[index]
        overlap = layer["mask"] & reference_mask
        gain, bias = robust_color_transform(layer["image"], reference, overlap)
        layer["gain"] = gain
        layer["bias"] = bias
        layer["normalized"] = apply_color_transform(layer["image"], gain, bias)

    before = mean_pairwise_difference([layer["image"] for layer in layers], [layer["mask"] for layer in layers])
    after = mean_pairwise_difference([layer["normalized"] for layer in layers], [layer["mask"] for layer in layers])

    scores = []
    for layer in layers:
        distance = cv2.distanceTransform(layer["mask"].astype(np.uint8), cv2.DIST_L2, 5)
        # Source quality dominates ownership. Distance from a semantic/occlusion edge is
        # deliberately only a tie-breaker, so the wall does not become a patchwork merely
        # because a weaker image happens to be farther from one of its mask boundaries.
        boundary_confidence = 0.95 + 0.05 * np.minimum(distance / 40.0, 1.0)
        scores.append(np.where(layer["mask"], layer["baseScore"] * boundary_confidence, 0))
    score_stack = np.stack(scores)
    owner_index = np.argmax(score_stack, axis=0)
    best_score = np.max(score_stack, axis=0)
    supported = best_score > 0
    ownership = np.where(supported, owner_index + 1, 0).astype(np.uint16)

    primary = np.zeros((height, width), dtype=np.uint16)
    support_count = np.zeros((height, width), dtype=np.uint16)
    all_valid = np.stack([layer["mask"] for layer in layers])
    support_count[:] = all_valid.sum(axis=0)
    primary[supported] = owner_index[supported] + 1
    raw_colours = np.stack([layer["image"] for layer in layers]).astype(np.float32)
    colours = np.stack([layer["normalized"] for layer in layers]).astype(np.float32)
    primary_colour = np.take_along_axis(
        colours,
        owner_index[None, :, :, None],
        axis=0,
    )[0]
    before_atlas = np.take_along_axis(
        raw_colours,
        owner_index[None, :, :, None],
        axis=0,
    )[0]
    before_atlas[~supported] = 0
    before_atlas = np.clip(before_atlas, 0, 255).astype(np.uint8)
    difference = np.mean(np.abs(colours - primary_colour[None]), axis=3)
    compatible = all_valid & (difference <= 25)
    layer_indices = np.arange(len(layers))[:, None, None]
    is_primary = layer_indices == owner_index[None]
    alternatives = compatible & ~is_primary
    alternative_scores = np.where(alternatives, score_stack, 0)
    alternative_total = alternative_scores.sum(axis=0)
    weights = np.zeros_like(score_stack, dtype=np.float32)
    has_alternative = alternative_total > 0
    weights += np.where(is_primary & supported[None], np.where(has_alternative, 0.75, 1.0), 0)
    weights += np.divide(
        alternative_scores * 0.25,
        alternative_total[None],
        out=np.zeros_like(alternative_scores, dtype=np.float32),
        where=alternative_total[None] > 0,
    )
    atlas = np.sum(colours * weights[:, :, :, None], axis=0)
    atlas = np.clip(atlas, 0, 255).astype(np.uint8)
    contributors = weights > 0
    contributor_count = contributors.sum(axis=0).astype(np.uint16)
    conflict = (all_valid & ~compatible).any(axis=0) & supported

    before_seam, _ = seam_magnitude(before_atlas, ownership)
    final_seam, final_seam_mask = seam_magnitude(atlas, ownership)
    ownership_distribution = {
        layers[index]["name"]: float((ownership == index + 1).sum() / max(1, supported.sum()) * 100)
        for index in range(len(layers))
    }
    metrics = {
        "preNormalizationMeanOverlapRgbDisagreement": before,
        "postNormalizationMeanOverlapRgbDisagreement": after,
        "beforeOwnershipSeamMeanRgbGradient": before_seam,
        "ownershipSeamMeanRgbGradient": final_seam,
        "ownershipSeamPixels": int(final_seam_mask.sum()),
        "atlasCoveragePercent": float(supported.mean() * 100),
        "multiViewAtlasPercent": float((support_count >= 2).mean() * 100),
        "singleViewAtlasPercent": float((support_count == 1).mean() * 100),
        "compatibleMultiContributorAtlasPercent": float((contributor_count >= 2).mean() * 100),
        "supportedAtlasConflictPercent": float(conflict.sum() / supported.sum() * 100),
        "ownershipDistributionPercent": ownership_distribution,
        "normalizedSources": [
            {
                "name": layer["name"],
                "gainBgr": layer["gain"].tolist(),
                "biasBgr": layer["bias"].tolist(),
                "baseScore": layer["baseScore"],
            }
            for layer in layers
        ],
    }
    return atlas, before_atlas, primary, contributor_count, weights, difference, compatible, metrics


def render_atlas(
    atlas: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    atlas_points: np.ndarray,
    plane_points: np.ndarray,
    target: View,
    camera,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_pixels, front = project_world(plane_points, target, camera)
    valid = (
        front
        & (target_pixels[:, 0] >= 0) & (target_pixels[:, 0] < camera.width)
        & (target_pixels[:, 1] >= 0) & (target_pixels[:, 1] < camera.height)
    )
    matrix, _, _ = homography(atlas_points[valid], target_pixels[valid])
    size = (camera.width, camera.height)
    colour = cv2.warpPerspective(atlas, matrix, size)
    owner = cv2.warpPerspective(primary, matrix, size, flags=cv2.INTER_NEAREST)
    second = cv2.warpPerspective(secondary, matrix, size, flags=cv2.INTER_NEAREST)
    provenance = np.full(owner.shape, UNSUPPORTED, dtype=np.uint8)
    provenance[(owner > 0) & (second == 0)] = CAPTURED_SINGLE_VIEW
    provenance[(owner > 0) & (second > 0)] = CAPTURED_MULTI_VIEW
    rgba = cv2.cvtColor(colour, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(owner > 0, 255, 0).astype(np.uint8)
    return rgba, provenance, owner, second


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
    parser.add_argument("--atlas-width", type=int, default=2400)
    parser.add_argument("--sweep-frames", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    args.output.mkdir(parents=True, exist_ok=True)
    camera = parse_camera(args.camera_text)
    views = parse_views(args.images_text)
    planes, models = detect_planes(args.ply, args.visibility, len(views))
    plane_points = models[0]["points"]
    plane_center, plane_normal, plane_axes = fit_plane(plane_points)
    threshold = planes[0]["distanceThreshold"]
    atlas_points, atlas_info = atlas_coordinates(
        plane_points,
        plane_center,
        plane_axes,
        args.atlas_width,
    )
    atlas_shape = (atlas_info["height"], atlas_info["width"])
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    mvs_names = [
        line.strip()
        for line in args.mvs_order.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("__") and "," not in line
    ]
    if len(mvs_names) != len(views) or set(mvs_names) != {view.name for view in views}:
        raise RuntimeError("MVS image order does not match registered views.")
    mvs_index = {name: index for index, name in enumerate(mvs_names)}

    layers = []
    rejected = []
    for view in views:
        source_id = mvs_index[view.name]
        visible_mask = np.array([source_id in support for support in models[0]["supports"]])
        visible_points = plane_points[visible_mask]
        visible_atlas = atlas_points[visible_mask]
        if len(visible_points) < 100:
            continue
        image = cv2.imread(str(args.images / view.name))
        depth = read_depth(args.depth_maps / f"{view.name}.geometric.bin")
        if image is None or image.shape[:2] != depth.shape or depth.shape != (camera.height, camera.width):
            raise RuntimeError(f"Image/depth dimensions disagree for {view.name}.")
        semantic = wall_mask(image, processor, model, torch)
        valid_depth, on_plane, off_plane = geometry_labels(
            depth, view, camera, plane_center, plane_normal, threshold
        )
        target_semantic, components, source_hull = target_wall_component(
            semantic, visible_points, view, camera
        )
        decisive = target_semantic & valid_depth & (on_plane | off_plane)
        recall = float((target_semantic & on_plane).sum() / max(1, (on_plane & source_hull).sum()) * 100)
        conflict = float((target_semantic & off_plane).sum() / max(1, target_semantic.sum()) * 100)
        agreement = float((decisive & on_plane).sum() / max(1, decisive.sum()) * 100)
        accepted = recall >= 20 and conflict <= 15 and agreement >= 55
        source_metric = {
            "name": view.name,
            "planeObservations": len(visible_points),
            "targetWallComponents": components,
            "geometryWallRecallPercent": recall,
            "geometryConflictPercent": conflict,
            "depthValidatedPlaneAgreementPercent": agreement,
            "accepted": accepted,
        }
        if not accepted:
            rejected.append(source_metric)
            continue
        refined = target_semantic & ~(
            cv2.dilate(off_plane.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        )
        source_pixels, front = project_world(visible_points, view, camera)
        valid = (
            front
            & (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < camera.width)
            & (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < camera.height)
        )
        matrix, rms, inliers = homography(source_pixels[valid], visible_atlas[valid])
        atlas_size = (atlas_info["width"], atlas_info["height"])
        warped_image = cv2.warpPerspective(image, matrix, atlas_size)
        warped_mask = cv2.warpPerspective(
            refined.astype(np.float32),
            matrix,
            atlas_size,
            flags=cv2.INTER_LINEAR,
        ) >= 1.0 - 1e-6
        plane_ray = plane_center - view.center
        distance = float(np.linalg.norm(plane_ray))
        plane_ray /= distance
        incidence = abs(float(np.dot(plane_ray, plane_normal)))
        source_area = max(1, int(refined.sum()))
        projected_density = math.sqrt(max(1, int(warped_mask.sum())) / source_area)
        base_score = incidence * projected_density * math.sqrt(inliers) / ((1 + rms) * (1 + 0.05 * distance))
        layers.append({
            "name": view.name,
            "image": warped_image,
            "mask": warped_mask,
            "baseScore": base_score,
            "metrics": source_metric,
        })

    if len(layers) < 2:
        raise RuntimeError("Fewer than two source views passed the wall evidence gate.")
    atlas, before_atlas, primary, contributor_count, weights, _, compatible, atlas_metrics = build_atlas(
        layers,
        atlas_shape,
    )
    atlas_alpha = np.where(primary > 0, 255, 0).astype(np.uint8)
    atlas_rgba = np.dstack((atlas, atlas_alpha))
    before_rgba = np.dstack((before_atlas, atlas_alpha))
    cv2.imwrite(str(args.output / "wall-atlas.png"), atlas_rgba)
    cv2.imwrite(str(args.output / "wall-atlas-before.png"), before_rgba)
    comparison = np.concatenate([before_rgba, atlas_rgba], axis=1)
    cv2.imwrite(str(args.output / "wall-atlas-before-after.png"), comparison)
    cv2.imwrite(str(args.output / "atlas-owner.png"), primary)
    cv2.imwrite(str(args.output / "atlas-contributor-count.png"), contributor_count)
    cv2.imwrite(
        str(args.output / "atlas-conflict.png"),
        np.where((np.stack([layer["mask"] for layer in layers]) & ~compatible).any(axis=0), 255, 0).astype(np.uint8),
    )
    np.savez_compressed(
        args.output / "atlas-source-weights.npz",
        weights=weights.astype(np.float16),
        source_names=np.asarray([layer["name"] for layer in layers]),
    )
    palette_hsv = np.zeros((1, len(layers) + 1, 3), dtype=np.uint8)
    palette_hsv[0, 1:, 0] = np.linspace(0, 179, len(layers), endpoint=False, dtype=np.uint8)
    palette_hsv[0, 1:, 1:] = 220
    palette = cv2.cvtColor(palette_hsv, cv2.COLOR_HSV2BGR)[0]
    cv2.imwrite(str(args.output / "atlas-owner-color.png"), palette[primary])
    atlas_provenance = np.zeros(primary.shape, dtype=np.uint8)
    atlas_provenance[(primary > 0) & (contributor_count == 1)] = CAPTURED_SINGLE_VIEW
    atlas_provenance[(primary > 0) & (contributor_count >= 2)] = CAPTURED_MULTI_VIEW
    cv2.imwrite(
        str(args.output / "atlas-provenance.png"),
        np.array([[0, 0, 0], [0, 165, 255], [0, 200, 0]], dtype=np.uint8)[atlas_provenance],
    )

    accepted_views = {layer["name"] for layer in layers}
    eligible = [view for view in views if view.name in accepted_views]
    first, second = min(
        ((a, b) for index, a in enumerate(eligible) for b in eligible[index + 1:]),
        key=lambda pair: np.linalg.norm(pair[0].center - pair[1].center)
        * (1 + math.radians(angle_degrees(pair[0].forward, pair[1].forward))),
    )
    sweep_dir = args.output / "sweep"
    sweep_dir.mkdir(exist_ok=True)
    render_metrics = []
    ownership_hashes = []
    for frame in range(args.sweep_frames):
        fraction = frame / (args.sweep_frames - 1)
        target = View(
            f"sweep-{frame:03}",
            quaternion_rotation(slerp(first.quaternion, second.quaternion, fraction)),
            first.center * (1 - fraction) + second.center * fraction,
            np.zeros(4),
            {},
        )
        rgba, provenance, owner, _ = render_atlas(
            atlas,
            primary,
            np.where(contributor_count >= 2, 1, 0).astype(np.uint8),
            atlas_points,
            plane_points,
            target,
            camera,
        )
        cv2.imwrite(str(sweep_dir / f"frame-{frame:03}.png"), rgba)
        cv2.imwrite(str(sweep_dir / f"owner-{frame:03}.png"), owner)
        ownership_hashes.append(hash(primary.tobytes()))
        visible = provenance > 0
        render_metrics.append({
            "frame": frame,
            "fraction": fraction,
            "directSourcePixelsPercent": float(visible.mean() * 100),
            "multiViewPercent": float((provenance == CAPTURED_MULTI_VIEW).mean() * 100),
            "singleViewPercent": float((provenance == CAPTURED_SINGLE_VIEW).mean() * 100),
            "unsupportedPercent": float((~visible).mean() * 100),
            "generatedPixels": 0,
        })
    frame_names = [f"sweep/frame-{frame:03}.png" for frame in range(args.sweep_frames)]
    sweep_html = """<!doctype html>
<meta charset="utf-8">
<title>Evidence-preserving wall sweep</title>
<style>
html,body { margin:0; height:100%; background:repeating-conic-gradient(#bbb 0 25%,#eee 0 50%) 0/24px 24px; }
img { display:block; width:100%; height:100%; object-fit:contain; }
</style>
<img id="frame" alt="Transparent evidence-preserving camera sweep">
<script>
const frames = __FRAMES__;
const image = document.getElementById("frame");
let index = 0;
function advance() {
  image.src = frames[index];
  index = (index + 1) % frames.length;
}
advance();
setInterval(advance, 1000 / 15);
</script>
""".replace("__FRAMES__", json.dumps(frame_names))
    (args.output / "camera-sweep.html").write_text(
        sweep_html,
        encoding="utf-8",
    )
    for index, fraction in enumerate((0.25, 0.5, 0.75), 1):
        frame = round(fraction * (args.sweep_frames - 1))
        for stem in ("frame", "owner"):
            source = sweep_dir / f"{stem}-{frame:03}.png"
            target = args.output / f"{stem}-view-{index}.png"
            target.write_bytes(source.read_bytes())

    result = {
        "model": args.model,
        "planePoints": len(plane_points),
        "atlas": atlas_info,
        "acceptedSources": [layer["metrics"] for layer in layers],
        "rejectedSources": rejected,
        "photographicCoherence": atlas_metrics,
        "sweep": {
            "frames": args.sweep_frames,
            "between": [first.name, second.name],
            "cameraDistance": float(np.linalg.norm(first.center - second.center)),
            "cameraAngularDistanceDegrees": angle_degrees(first.forward, second.forward),
            "canonicalOwnershipSwitches": len(set(ownership_hashes)) - 1,
            "renders": render_metrics,
        },
        "provenance": {
            "primaryOwnerRaster": "atlas-owner.png",
            "contributorCountRaster": "atlas-contributor-count.png",
            "sourceWeightArchive": "atlas-source-weights.npz",
            "generatedPixels": 0,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote coherent wall atlas and {args.sweep_frames}-frame sweep to {args.output}")


if __name__ == "__main__":
    main()
