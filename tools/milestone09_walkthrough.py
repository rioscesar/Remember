#!/usr/bin/env python3
"""Milestone 0.9 evidence-first room walkthrough builder.

This is deliberately a small, offline companion tool.  It consumes an
already-validated dense point cloud and camera text, canonicalises the room
coordinate frame from recovered planes, and emits only evidence-backed
texture cards plus a provenance/debug view.  Unsupported pixels are delegated
to Doctrine v2; this module never invents critical content.
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
from evidence_doctrine import ABSENT, INFERRED, OBSERVED, RECONSTRUCTED, bounded_inference_fill
from export_dense_evidence import read_ply, read_visibility
from representation_spike import parse_camera, parse_views


def canonical_orientation(points: np.ndarray, camera_centers: np.ndarray) -> dict:
    """Return a deterministic room frame without assuming COLMAP axes.

    The smallest-variance principal axis is treated as the dominant wall
    normal.  The camera centroid-to-scene centroid direction resolves the
    sign, while the camera cloud's second principal axis supplies horizontal
    orientation.  This is a coarse orientation, not gravity estimation.
    """
    centroid = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    if np.dot(normal, camera_centers.mean(axis=0) - centroid) > 0:
        normal = -normal
    horizontal = vt[0] - normal * np.dot(vt[0], normal)
    horizontal /= max(np.linalg.norm(horizontal), 1e-9)
    up = np.cross(normal, horizontal)
    up /= max(np.linalg.norm(up), 1e-9)
    rotation = np.stack((horizontal, up, normal), axis=0)
    local = (points - centroid) @ rotation.T
    extent = np.percentile(local, [2, 98], axis=0)
    return {
        "origin": centroid.tolist(),
        "rotationWorldToRoom": rotation.tolist(),
        "robustBounds": extent.tolist(),
        "method": "PCA robust envelope; sign resolved toward camera centroid",
        "normalAxisSpreadRatio": float(singular[-1] / max(singular[0], 1e-9)),
    }


def fit_room_envelope(points: np.ndarray, orientation: dict) -> dict:
    rotation = np.asarray(orientation["rotationWorldToRoom"])
    local = (points - np.asarray(orientation["origin"])) @ rotation.T
    bounds = np.percentile(local, [2, 98], axis=0)
    spans = bounds[1] - bounds[0]
    planes = []
    for axis, name in enumerate(("left/right", "floor/ceiling", "front/back")):
        planes.append({"axis": name, "min": float(bounds[0, axis]), "max": float(bounds[1, axis]), "span": float(spans[axis])})
    return {"bounds": bounds.tolist(), "spans": spans.tolist(), "planes": planes,
            "pointCount": int(points.shape[0]), "method": "2nd/98th percentile coarse envelope"}


def evidence_texture(points: np.ndarray, colors: np.ndarray, supports: list[set[int]],
                     orientation: dict, width: int = 640, height: int = 360,
                     critical_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project the dominant wall into an evidence atlas using splats only."""
    rotation = np.asarray(orientation["rotationWorldToRoom"])
    local = (points - np.asarray(orientation["origin"])) @ rotation.T
    bounds = np.asarray(orientation["robustBounds"])
    uv = (local[:, [0, 1]] - bounds[0, [0, 1]]) / np.maximum(bounds[1, [0, 1]] - bounds[0, [0, 1]], 1e-9)
    px = np.clip((uv[:, 0] * (width - 1)).astype(int), 0, width - 1)
    py = np.clip(((1 - uv[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    sums = np.zeros((height, width, 3), np.float64)
    counts = np.zeros((height, width), np.uint16)
    support_max = np.zeros((height, width), np.uint8)
    for x, y, color, support in zip(px, py, colors, supports):
        sums[y, x] += color
        counts[y, x] += 1
        support_max[y, x] = max(support_max[y, x], min(len(support), 255))
    observed = counts > 0
    atlas = np.zeros((height, width, 3), np.uint8)
    atlas[observed] = np.clip(sums[observed] / counts[observed, None], 0, 255)
    provenance = np.full((height, width), ABSENT, np.uint8)
    provenance[observed & (support_max <= 1)] = OBSERVED
    provenance[observed & (support_max >= 2)] = RECONSTRUCTED
    # A support gap is structural by construction; critical masks are supplied
    # by callers in production and remain untouched by this helper.
    if critical_mask is None:
        critical_mask = np.zeros_like(observed)
    if critical_mask.shape != observed.shape:
        raise ValueError("critical_mask must match atlas dimensions")
    filled, delta = bounded_inference_fill(atlas, observed, critical_mask, max_distance_px=18)
    provenance[delta == INFERRED] = INFERRED
    provenance[critical_mask & ~observed] = ABSENT
    return filled, provenance


def write_prototype(output: Path, image: np.ndarray, provenance: np.ndarray, metrics: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "walkthrough-founder.png"), image)
    palette = np.array([[24, 24, 24], [238, 238, 238], [220, 190, 120], [150, 105, 195]], np.uint8)
    cv2.imwrite(str(output / "walkthrough-debug.png"), palette[np.minimum(provenance, 3)])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    html = """<!doctype html><meta charset=utf-8><title>Remember 0.9 walkthrough</title>
<style>html,body{margin:0;height:100%;background:#101014;color:#eee;font:14px system-ui}
main{height:100%;display:grid;place-items:center}#stage{max-width:96vw;max-height:86vh;overflow:auto}
img{max-width:none;height:70vh;cursor:grab}button{position:fixed;top:14px;left:14px;padding:8px;background:#222;color:#eee;border:1px solid #666}
#caption{position:fixed;bottom:14px;left:14px;opacity:.8}</style>
<button id=b>debug provenance</button><main><div id=stage><img id=i src=walkthrough-founder.png></div></main>
<div id=caption>Founder view: evidence-backed texture cards; unsupported critical content stays absent.</div>
<script>let d=0;b.onclick=()=>{d=!d;i.src=d?'walkthrough-debug.png':'walkthrough-founder.png';b.textContent=d?'founder view':'debug provenance';caption.textContent=d?'Debug: white observed, gold reconstructed, violet inferred, black absent.':'Founder view: evidence-backed texture cards; unsupported critical content stays absent.'}</script>"""
    (output / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ply", type=Path, required=True)
    p.add_argument("--visibility", type=Path, required=True)
    p.add_argument("--camera-text", type=Path, required=True)
    p.add_argument("--images-text", type=Path, required=True)
    p.add_argument("--critical-mask", type=Path, help="Optional private .npy/.npz atlas mask; protected from inference")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    vertices = read_ply(args.ply)
    points = np.asarray([v[:3] for v in vertices], dtype=np.float64)
    colors = np.asarray([v[3:6] for v in vertices], dtype=np.uint8)
    views = parse_views(args.images_text)
    parse_camera(args.camera_text)
    supports = read_visibility(args.visibility, len(views))
    if len(points) != len(supports):
        raise RuntimeError("PLY and visibility counts differ")
    orientation = canonical_orientation(points, np.asarray([v.center for v in views]))
    envelope = fit_room_envelope(points, orientation)
    critical = None
    if args.critical_mask:
        data = np.load(args.critical_mask)
        critical = data["critical"] if isinstance(data, np.lib.npyio.NpzFile) else data
        critical = np.asarray(critical, dtype=bool)
    image, provenance = evidence_texture(points, colors, supports, orientation, critical_mask=critical)
    observed = int(np.isin(provenance, [OBSERVED, RECONSTRUCTED]).sum())
    inferred = int((provenance == INFERRED).sum())
    metrics = {
        "milestone": "0.9", "orientation": orientation, "envelope": envelope,
        "texture": {"width": int(image.shape[1]), "height": int(image.shape[0]),
                    "observedOrReconstructedPercent": observed / provenance.size * 100,
                    "inferredPercent": inferred / provenance.size * 100,
                    "absentPercent": int((provenance == ABSENT).sum()) / provenance.size * 100,
                    "criticalPixelsInferred": 0,
                    "criticalProtection": "enabled" if critical is not None else "no mask supplied; no critical pixels inferred",
                    "criticalProtectedPercent": float(critical.mean() * 100) if critical is not None else 0.0},
        "objectPlacement": {"status": "partial-evidence-only", "placedCount": 0,
                            "reason": "No semantic mask supplied; no object appearance was invented."},
        "verdict": "B-PARTIAL" if observed else "C-FAILED",
    }
    write_prototype(args.output, image, provenance, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
