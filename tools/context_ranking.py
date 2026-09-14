"""Contextual memory ranking and source photo selection for Remember Milestone 1.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass
class RankedSourcePhoto:
    name: str
    view_index: int
    overall_score: float
    geometry_score: float
    spatial_score: float
    alignment_score: float
    semantic_score: float
    wall_points: int
    total_points: int
    distance_to_origin: float
    wall_fraction: float
    camera_pos_room: list[float]
    camera_dir_room: list[float]


def rank_source_photos(
    views: Sequence,
    supports: Sequence[set[int] | list[int]],
    wall_supports: Sequence[set[int] | list[int]],
    orientation: dict,
    img_dir: Path,
    target_face_normal_room: tuple[float, float, float] = (0.0, 0.0, -1.0),
    top_k: int = 4,
    processor=None,
    segmentation_model=None,
    device: str = "cuda",
) -> list[RankedSourcePhoto]:
    """Rank source photographs using camera, geometry, semantic, and spatial signals.

    Contextual evidence from ranked photos may guide generation (e.g. style,
    palette, material features), but must always remain IMAGINED when applied
    to unobserved target regions, never relabeled as RECONSTRUCTED.
    """
    rotation = np.asarray(orientation["rotationWorldToRoom"])
    origin = np.asarray(orientation["origin"])
    target_normal = np.asarray(target_face_normal_room, dtype=np.float64)
    target_norm_len = np.linalg.norm(target_normal)
    if target_norm_len > 1e-9:
        target_normal = target_normal / target_norm_len

    raw_candidates = []
    max_wall_pts = 1
    max_total_pts = 1

    for idx, view in enumerate(views):
        img_path = img_dir / view.name
        if not img_path.exists():
            continue

        wall_pts = sum(1 for s in wall_supports if idx in s)
        tot_pts = sum(1 for s in supports if idx in s)
        max_wall_pts = max(max_wall_pts, wall_pts)
        max_total_pts = max(max_total_pts, tot_pts)

        cam_pos_room = rotation @ (view.center - origin)
        cam_dir_world = view.rotation.T @ np.array([0.0, 0.0, 1.0])
        cam_dir_room = rotation @ cam_dir_world
        dist = float(np.linalg.norm(cam_pos_room))

        # Semantic wall fraction if segmentation model provided
        wall_frac = 0.35  # default estimate
        if processor is not None and segmentation_model is not None:
            try:
                import torch
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    h, w = img_bgr.shape[:2]
                    scale = 512 / max(h, w)
                    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
                    inputs = processor(images=cv2.cvtColor(small, cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = segmentation_model(**inputs)
                        pred = torch.nn.functional.interpolate(
                            out.logits, size=small.shape[:2], mode="bilinear", align_corners=False
                        ).argmax(dim=1)[0].cpu().numpy()
                    wall_frac = float((pred == 0).mean())
            except Exception:
                pass

        raw_candidates.append({
            "name": view.name,
            "index": idx,
            "wall_pts": wall_pts,
            "tot_pts": tot_pts,
            "pos_room": cam_pos_room,
            "dir_room": cam_dir_room,
            "dist": dist,
            "wall_frac": wall_frac,
        })

    ranked: list[RankedSourcePhoto] = []
    for cand in raw_candidates:
        geom_score = (cand["wall_pts"] / max_wall_pts) * 0.7 + (cand["tot_pts"] / max_total_pts) * 0.3
        spatial_score = float(np.exp(-cand["dist"] / 8.0))
        # Alignment: dot product with view direction / coverage
        align_score = float(np.clip(np.dot(cand["dir_room"], -target_normal) * 0.5 + 0.5, 0.0, 1.0))
        semantic_score = float(np.clip(cand["wall_frac"] / 0.5, 0.0, 1.0))

        # Weighted combination: geometry (0.45), spatial (0.20), semantic (0.20), alignment (0.15)
        overall = (
            0.45 * geom_score
            + 0.20 * spatial_score
            + 0.20 * semantic_score
            + 0.15 * align_score
        )

        ranked.append(RankedSourcePhoto(
            name=cand["name"],
            view_index=cand["index"],
            overall_score=round(float(overall), 4),
            geometry_score=round(float(geom_score), 4),
            spatial_score=round(float(spatial_score), 4),
            alignment_score=round(float(align_score), 4),
            semantic_score=round(float(semantic_score), 4),
            wall_points=int(cand["wall_pts"]),
            total_points=int(cand["tot_pts"]),
            distance_to_origin=round(float(cand["dist"]), 3),
            wall_fraction=round(float(cand["wall_frac"]), 4),
            camera_pos_room=[round(float(x), 3) for x in cand["pos_room"]],
            camera_dir_room=[round(float(x), 3) for x in cand["dir_room"]],
        ))

    ranked.sort(key=lambda x: x.overall_score, reverse=True)
    return ranked[:top_k]


def build_context_contact_sheet(
    ranked_photos: list[RankedSourcePhoto],
    img_dir: Path,
    crop_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Build a private context contact sheet and composite reference grid from ranked source photos.

    Returns:
      (contact_sheet_bgr, crops_bgr_list, palette_info)
    """
    crops_bgr = []
    labels = []
    palette_samples = []

    for item in ranked_photos:
        p = img_dir / item.name
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Structural wall region crop (central upper-middle region)
        crop = img[int(h * 0.2):int(h * 0.75), int(w * 0.2):int(w * 0.8)]
        crop_resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA)
        crops_bgr.append(crop_resized)
        labels.append(f"{item.name} (score: {item.overall_score:.2f})")
        palette_samples.append(crop_resized.reshape(-1, 3))

    if not crops_bgr:
        empty = np.full((crop_size[1], crop_size[0], 3), 128, dtype=np.uint8)
        return empty, [empty], {"meanColorBgr": [128, 128, 128]}

    # Layout into 2x2 or 1xN grid
    n = len(crops_bgr)
    if n >= 4:
        r1 = np.hstack((crops_bgr[0], crops_bgr[1]))
        r2 = np.hstack((crops_bgr[2], crops_bgr[3]))
        grid = np.vstack((r1, r2))
    elif n == 3:
        r1 = np.hstack((crops_bgr[0], crops_bgr[1]))
        r2 = np.hstack((crops_bgr[2], np.zeros_like(crops_bgr[2])))
        grid = np.vstack((r1, r2))
    elif n == 2:
        grid = np.hstack((crops_bgr[0], crops_bgr[1]))
    else:
        grid = crops_bgr[0]

    all_px = np.vstack(palette_samples)
    mean_bgr = all_px.mean(axis=0).tolist()
    std_bgr = all_px.std(axis=0).tolist()
    lab = cv2.cvtColor(all_px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    mean_lab = lab.mean(axis=0).flatten().tolist()

    palette_info = {
        "meanColorBgr": [round(x, 2) for x in mean_bgr],
        "stdColorBgr": [round(x, 2) for x in std_bgr],
        "meanColorLab": [round(x, 2) for x in mean_lab],
        "rankedCount": len(ranked_photos),
        "sourcePhotos": [p.name for p in ranked_photos],
    }

    # Annotate contact sheet with photo names and scores
    sheet_h, sheet_w = grid.shape[:2]
    annotated = cv2.resize(grid, (sheet_w, sheet_h))
    header = np.zeros((36, sheet_w, 3), dtype=np.uint8)
    cv2.putText(
        header,
        f"Contextual Memory Source Photos (Ranked {len(ranked_photos)}) - IMAGINED Context",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    contact_sheet = np.vstack((header, annotated))

    return contact_sheet, crops_bgr, palette_info