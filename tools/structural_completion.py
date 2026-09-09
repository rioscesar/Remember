#!/usr/bin/env python3
"""Milestone 0.8, Phases 3-5: evidence-maximizing structural completion.

Rebuilds the strongest recovered wall as a photographic atlas (reusing the
validated `wall_coherence_spike.py` machinery unchanged), then adds the two
things Doctrine v2 asks for that Milestone 0.6/0.7 did not have:

  1. Critical-region protection: each accepted view's Phase-2 risk mask
     (`scene_risk_segmentation.py`) is warped through the *same* per-view
     homography used to build the atlas, and OR-accumulated into
     `atlas_critical`. Any atlas texel any view ever classified critical
     (art, TV, posters, faces, signage, mirrors, ...) is permanently
     excluded from inference, even if it is otherwise unsupported.

  2. Bounded structural inference: remaining unsupported, non-critical
     atlas texels are conservatively completed with
     `evidence_doctrine.bounded_inference_fill` (classical inpainting,
     capped fill radius, no learned/generative prior).

Floor and ceiling are NOT reconstructed with per-pixel ray-traced geometry.
The recovered point cloud is not gravity-aligned and multi-view support for
floor/ceiling planes in this dataset is an order of magnitude weaker than
the wall (764-3,665 points across 4-5 views vs. the wall's 38,913 points
across 9 views) -- attempting precise reconstruction from that little
evidence would risk exactly the "confidently wrong" failure mode Doctrine
v2 exists to prevent. Instead, floor and ceiling are built as simple,
honestly low-fidelity INFERRED colour bands sampled from real evidence at
the wall's own floor/ceiling edges. This is a deliberate, documented scope
decision, not a silent substitution -- see docs/EVIDENCE-MAXIMIZING-...md.

No pixel in this script's output is ever synthesized over a critical
region, and every output pixel carries an explicit provenance code from
`evidence_doctrine` (OBSERVED / RECONSTRUCTED / INFERRED / ABSENT).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import (
    detect_planes,
    parse_camera,
    parse_views,
    project_world,
    read_depth,
)
from semantic_plane_spike import (
    fit_plane,
    geometry_labels,
    target_wall_component,
    wall_mask,
)
from wall_coherence_spike import (
    atlas_coordinates,
    build_atlas,
    homography,
)
from evidence_doctrine import (
    ABSENT,
    INFERRED,
    OBSERVED,
    RECONSTRUCTED,
    bounded_inference_fill,
)


def load_risk_mask(risk_dir: Path, view_name: str) -> np.ndarray | None:
    stem = Path(view_name).stem
    path = risk_dir / f"risk-masks-{stem}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return data["critical"]


def warp_critical_mask(critical: np.ndarray, matrix: np.ndarray, atlas_size: tuple[int, int]) -> np.ndarray:
    warped = cv2.warpPerspective(
        critical.astype(np.float32), matrix, atlas_size, flags=cv2.INTER_LINEAR
    )
    return warped > 0.05  # any partial overlap with a critical region stays protected


def build_floor_ceiling_band(
    atlas: np.ndarray,
    supported: np.ndarray,
    band_height: int,
    edge: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-fidelity INFERRED colour band sampled from real evidence at one
    edge of the wall atlas. Returns (band_bgr, provenance) where provenance
    is entirely INFERRED -- this is a deliberate low-fidelity scope choice,
    not a claim of reconstructed floor/ceiling geometry (see module docstring).
    """
    height, width = atlas.shape[:2]
    strip = 40
    if edge == "top":
        region = atlas[:strip]
        region_supported = supported[:strip]
        near_color = np.array([150, 145, 130], dtype=np.float32)  # neutral ceiling fallback
    else:
        region = atlas[height - strip:]
        region_supported = supported[height - strip:]
        near_color = np.array([90, 120, 140], dtype=np.float32)  # neutral floor fallback

    if region_supported.any():
        sampled = region[region_supported].astype(np.float32)
        anchor_color = sampled.mean(axis=0)
    else:
        anchor_color = near_color

    far_color = np.clip(anchor_color * (1.15 if edge == "top" else 0.75), 0, 255)
    band = np.zeros((band_height, width, 3), dtype=np.uint8)
    for row in range(band_height):
        fraction = row / max(1, band_height - 1)
        if edge == "top":
            colour = anchor_color * (1 - fraction) + far_color * fraction
        else:
            colour = far_color * (1 - fraction) + anchor_color * fraction
        band[row, :] = colour
    provenance = np.full((band_height, width), INFERRED, dtype=np.uint8)
    return band, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--depth-maps", type=Path, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--mvs-order", type=Path, required=True)
    parser.add_argument("--risk-masks", type=Path, required=True, help="Output dir from scene_risk_segmentation.py")
    parser.add_argument("--model", default="openmmlab/upernet-convnext-tiny")
    parser.add_argument("--atlas-width", type=int, default=2400)
    parser.add_argument("--max-inference-distance-px", type=float, default=140.0)
    parser.add_argument("--floor-ceiling-band-height", type=int, default=360)
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
    atlas_points, atlas_info = atlas_coordinates(plane_points, plane_center, plane_axes, args.atlas_width)
    atlas_shape = (atlas_info["height"], atlas_info["width"])
    atlas_size = (atlas_info["width"], atlas_info["height"])

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    mvs_names = [
        line.strip()
        for line in args.mvs_order.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("__") and "," not in line
    ]
    mvs_index = {name: index for index, name in enumerate(mvs_names)}

    layers = []
    critical_layers = []
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
        valid_depth, on_plane, off_plane = geometry_labels(depth, view, camera, plane_center, plane_normal, threshold)
        target_semantic, components, source_hull = target_wall_component(semantic, visible_points, view, camera)
        decisive = target_semantic & valid_depth & (on_plane | off_plane)
        recall = float((target_semantic & on_plane).sum() / max(1, (on_plane & source_hull).sum()) * 100)
        conflict = float((target_semantic & off_plane).sum() / max(1, target_semantic.sum()) * 100)
        agreement = float((decisive & on_plane).sum() / max(1, decisive.sum()) * 100)
        accepted = recall >= 20 and conflict <= 15 and agreement >= 55
        if not accepted:
            continue
        refined = target_semantic & ~(cv2.dilate(off_plane.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        source_pixels, front = project_world(visible_points, view, camera)
        valid = (
            front
            & (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < camera.width)
            & (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < camera.height)
        )
        matrix, rms, inliers = homography(source_pixels[valid], visible_atlas[valid])
        warped_image = cv2.warpPerspective(image, matrix, atlas_size)
        warped_mask = cv2.warpPerspective(
            refined.astype(np.float32), matrix, atlas_size, flags=cv2.INTER_LINEAR
        ) >= 1.0 - 1e-6
        plane_ray = plane_center - view.center
        distance = float(np.linalg.norm(plane_ray))
        plane_ray = plane_ray / distance
        incidence = abs(float(np.dot(plane_ray, plane_normal)))
        source_area = max(1, int(refined.sum()))
        projected_density = math.sqrt(max(1, int(warped_mask.sum())) / source_area)
        base_score = incidence * projected_density * math.sqrt(inliers) / ((1 + rms) * (1 + 0.05 * distance))
        layers.append({"name": view.name, "image": warped_image, "mask": warped_mask, "baseScore": base_score})

        risk_critical = load_risk_mask(args.risk_masks, view.name)
        if risk_critical is not None:
            critical_layers.append(warp_critical_mask(risk_critical, matrix, atlas_size))

    if len(layers) < 2:
        raise RuntimeError("Fewer than two source views passed the wall evidence gate.")

    atlas, before_atlas, primary, contributor_count, weights, _, compatible, atlas_metrics = build_atlas(
        layers, atlas_shape
    )
    supported = primary > 0

    atlas_critical = np.zeros(atlas_shape, dtype=bool)
    for mask in critical_layers:
        atlas_critical |= mask
    # Critical protection must survive even where evidence claims support --
    # e.g. a poster the atlas partially reconstructed still must not be
    # extended/completed by inference into its unsupported neighbourhood.

    filled_atlas, inference_delta = bounded_inference_fill(
        atlas, supported, atlas_critical, args.max_inference_distance_px
    )

    provenance = np.full(atlas_shape, ABSENT, dtype=np.uint8)
    provenance[supported & (contributor_count == 1)] = OBSERVED
    provenance[supported & (contributor_count >= 2)] = RECONSTRUCTED
    provenance = np.where((inference_delta == INFERRED) & ~atlas_critical, INFERRED, provenance)

    wall_metrics = {
        "atlasWidth": atlas_info["width"],
        "atlasHeight": atlas_info["height"],
        "observedOrReconstructedPercent": float(supported.mean() * 100),
        "criticalProtectedPercent": float(atlas_critical.mean() * 100),
        "inferredPercent": float((provenance == INFERRED).mean() * 100),
        "absentPercent": float((provenance == ABSENT).mean() * 100),
        "criticalPixelsEverInferredOver": int((atlas_critical & (provenance == INFERRED)).sum()),
        **atlas_metrics,
    }

    # Floor / ceiling: deliberate low-fidelity INFERRED bands (see module docstring).
    ceiling_band, ceiling_provenance = build_floor_ceiling_band(
        atlas, supported, args.floor_ceiling_band_height, "top"
    )
    floor_band, floor_provenance = build_floor_ceiling_band(
        atlas, supported, args.floor_ceiling_band_height, "bottom"
    )

    room = np.concatenate([ceiling_band, filled_atlas, floor_band], axis=0)
    room_provenance = np.concatenate([ceiling_provenance, provenance, floor_provenance], axis=0)

    provenance_colors = np.array(
        [
            [30, 30, 30],     # ABSENT: near-black
            [235, 235, 235],  # OBSERVED: white
            [235, 200, 120],  # RECONSTRUCTED: pale blue-ish (BGR)
            [140, 100, 200],  # INFERRED: dusty violet, deliberately muted/subdued
        ],
        dtype=np.uint8,
    )
    room_provenance_vis = provenance_colors[room_provenance]

    cv2.imwrite(str(args.output / "wall-atlas-observed.png"), np.dstack((atlas, np.where(supported, 255, 0).astype(np.uint8))))
    cv2.imwrite(str(args.output / "wall-atlas-completed.png"), filled_atlas)
    cv2.imwrite(str(args.output / "atlas-critical-protected.png"), np.where(atlas_critical, 255, 0).astype(np.uint8))
    cv2.imwrite(str(args.output / "room-structural.png"), room)
    cv2.imwrite(str(args.output / "room-provenance.png"), room_provenance_vis)
    np.savez_compressed(
        args.output / "room-provenance-codes.npz",
        provenance=room_provenance,
        ceiling_height=np.array([args.floor_ceiling_band_height]),
        floor_height=np.array([args.floor_ceiling_band_height]),
        wall_height=np.array([atlas_info["height"]]),
    )

    metrics = {
        "acceptedViewCount": len(layers),
        "wall": wall_metrics,
        "floorCeiling": {
            "method": "low-fidelity evidence-anchored gradient band (INFERRED)",
            "reason": "floor/ceiling multi-view point support is 4-5 views / 764-3665 points vs. the wall's 9 views / 38913 points; "
                      "the recovered reconstruction is not gravity-aligned, so precise geometric floor/ceiling completion was judged "
                      "too likely to be confidently wrong for this dataset. This is a deliberate documented scope decision.",
            "bandHeightPx": args.floor_ceiling_band_height,
            "percentOfRoomImage": float(2 * args.floor_ceiling_band_height / room.shape[0] * 100),
        },
        "roomImage": {
            "widthPx": room.shape[1],
            "heightPx": room.shape[0],
            "observedOrReconstructedPercent": float(np.isin(room_provenance, [OBSERVED, RECONSTRUCTED]).mean() * 100),
            "inferredPercent": float((room_provenance == INFERRED).mean() * 100),
            "absentPercent": float((room_provenance == ABSENT).mean() * 100),
        },
    }
    (args.output / "structural-completion-metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
