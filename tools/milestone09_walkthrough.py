#!/usr/bin/env python3
"""Milestone 1.0 evidence-first Remember/Imagine room-shell builder.

This is a small, offline companion tool with no WebGL/Three.js dependency.
It consumes an already-validated dense point cloud (PLY + MVS visibility
sidecar) and camera poses, then:

  1. Canonicalises a deterministic room coordinate frame from recovered
     geometry (`canonical_orientation`). This is explicitly NOT gravity
     estimation -- see the "Orientation limitation" note below.
  2. Fits a coarse robust room envelope (`fit_room_envelope`).
  3. Detects up to five dominant structural planes
     (`representation_spike.detect_planes`) and classifies each as a
     candidate floor/ceiling/left/right/front/back face of a coarse box
     shell (`classify_face`), reusing the SAME plane-detection code path
     validated in Milestones 0.3/0.8 rather than re-deriving geometry here.
  4. Builds a real evidence-splat texture card in canonical room-face/atlas
     space for every face that has a
     matching recovered plane (`splat_atlas`) -- OBSERVED/RECONSTRUCTED
     pixels come directly from real per-point colour; unsupported,
     non-critical gaps are bounded-inferred exactly as Doctrine v2 requires
     (`evidence_doctrine.bounded_inference_fill`). Faces with NO recovered
     plane are completed first by the local deterministic Imagine fallback:
     a generic structural atlas card marked 100% IMAGINED, visible only when
     the user turns on Imagine. This is not frame-by-frame generation and is
     not represented as evidence.
  5. Clusters the residual (non-planar) points left over after plane
     removal into candidate furniture/object volumes
     (`cluster_residual_points`) and places each as a flat, explicitly
     labelled evidence card at its REAL measured 3D position/size/colour --
     partial evidence-backed object placement with no shape completion.
  6. Emits a genuine pannable/walkable 3D prototype
     (`write_prototype`/`SHELL_TEMPLATE`) using pure CSS 3D transforms
     (perspective + preserve-3d + per-face transforms) -- no canvas/WebGL,
     no Three.js -- with pointer-drag look + WASD/arrow-key camera
     translation clamped to the recovered envelope, a Remember/Imagine
     toggle, and a debug provenance view that swaps every face and object
     card's texture.

## Orientation limitation (explicitly documented, not hidden)

`canonical_orientation` derives a room frame purely from the recovered
point cloud's principal axes and the camera-centroid direction. It has NO
access to gravity: this pipeline is 100% offline COLMAP Structure-from-
Motion over ordinary photographs, with no IMU/accelerometer stream to align
"up" against true gravity. The "up" axis produced here is therefore a
best-effort geometric proxy (typically close to true up when a dominant
vertical wall was recovered, because that wall's plane normal is
horizontal and the wall's own vertical extent dominates local axis 1) --
it is not validated against gravity and must not be described as such.
Fixing this would require either (a) capturing device IMU/gravity data
alongside the photographs (an Android/AR capture change, out of scope this
milestone) or (b) a dedicated vertical-vanishing-point estimation spike.
Both are explicitly deferred; this is a measured, reported blocker, not a
silent omission.

## Scale limitation (explicitly documented, not hidden)

Ordinary-photo SfM has arbitrary scale (see docs/ARCHITECTURE.md, "No
distance is displayed"). Every CSS pixel size in the generated shell is a
fixed visualization multiplier (`PX_PER_UNIT`) over the reconstruction's
own unitless coordinates -- it is a shape/proportion prototype, not a
measured-distance walkthrough.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import (
    ABSENT,
    IMAGINED,
    INFERRED,
    OBSERVED,
    RECONSTRUCTED,
    apply_provenance_priority,
    bounded_inference_fill,
    generation_masks,
    provenance_percentages,
)
from export_dense_evidence import read_ply, read_visibility
from representation_spike import detect_planes, parse_views
from semantic_plane_spike import fit_plane

PX_PER_UNIT = 130  # fixed visualization scale; reconstruction units are unitless (arbitrary SfM scale).
FACE_KEYS = ("left", "right", "floor", "ceiling", "back", "front")
# axis: which local room axis (0=horizontal,1=up,2=depth) this face lies
# perpendicular to; sign: which side of room centre this face sits on.
FACE_AXIS = {"left": (0, -1), "right": (0, 1), "floor": (1, -1), "ceiling": (1, 1),
             "back": (2, -1), "front": (2, 1)}
MIN_PLANE_POINTS_FOR_FACE = 150
MIN_RESIDUAL_CLUSTER_POINTS = 40
MAX_GENERATED_OBJECT_GAPS = 2


def canonical_orientation(points: np.ndarray, camera_centers: np.ndarray) -> dict:
    """Return a deterministic room frame without assuming COLMAP axes.

    The smallest-variance principal axis is treated as the dominant wall
    normal.  The camera centroid-to-scene centroid direction resolves the
    sign, while the camera cloud's second principal axis supplies horizontal
    orientation.  This is a coarse geometric-proxy orientation, NOT gravity
    estimation -- see the module docstring's "Orientation limitation".
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
        "gravityEstimated": False,
        "gravityBlockedReason": "No IMU/gravity sensor stream available in this offline COLMAP-only pipeline; "
                                 "see module docstring 'Orientation limitation'.",
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


def splat_atlas(points: np.ndarray, colors: np.ndarray, supports: list, width: int = 480) -> tuple[np.ndarray, np.ndarray, dict]:
    """Evidence-only per-point colour splat atlas in this plane's own best-fit 2D frame.

    Reuses `semantic_plane_spike.fit_plane` (validated Milestone 0.3+) for the
    local 2D axes so texture quality is not distorted by the room's global
    frame. No photograph is read here -- colour comes directly from the
    fused-point RGB already carried by the dense evidence PLY.
    """
    center, _, axes = fit_plane(points)
    coordinates = (points - center) @ axes.T
    low = np.percentile(coordinates, 2, axis=0)
    high = np.percentile(coordinates, 98, axis=0)
    span = np.maximum(high - low, 1e-9)
    height = max(1, round(width * span[1] / span[0]))
    height = min(height, 2000)
    uv = (coordinates - low) / span
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
    critical = np.zeros_like(observed)  # no per-view semantic risk mask available from a point-only PLY input.
    filled, delta = bounded_inference_fill(atlas, observed, critical, max_distance_px=max(width, height) * 0.05)
    candidate_provenance = provenance.copy()
    candidate_provenance[delta == INFERRED] = INFERRED
    filled, provenance = apply_provenance_priority(
        atlas,
        provenance,
        filled,
        candidate_provenance,
        locked_mask=np.isin(provenance, [OBSERVED, RECONSTRUCTED]),
    )
    info = {
        "widthPx": width, "heightPx": height,
        "observedOrReconstructedPercent": float(np.isin(provenance, [OBSERVED, RECONSTRUCTED]).mean() * 100),
        **provenance_percentages(provenance),
    }
    return filled, provenance, info


def deterministic_structural_imagine(
    width: int,
    height: int,
    seed_color_bgr: np.ndarray,
    generatable_mask: np.ndarray,
    locked_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Local deterministic fallback for missing structural room faces.

    This is deliberately generic and atlas-space only: a muted structural
    gradient seeded from recovered face colours. It never reads source
    photographs, never runs cloud inference, and never paints locked pixels.
    """
    width = max(1, int(width))
    height = max(1, int(height))
    if generatable_mask.shape != (height, width) or locked_mask.shape != (height, width):
        raise ValueError("generation masks must match requested atlas size")

    y, x = np.mgrid[0:height, 0:width]
    gradient = ((x / max(width - 1, 1)) * 10 + (y / max(height - 1, 1)) * 7).astype(np.float32)
    grain = (((x * 37 + y * 17) % 11) - 5).astype(np.float32)
    base = np.clip(seed_color_bgr.astype(np.float32), 72, 190)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = np.clip(base + gradient[:, :, None] + grain[:, :, None], 0, 255).astype(np.uint8)
    provenance = np.full((height, width), ABSENT, dtype=np.uint8)
    provenance[generatable_mask & ~locked_mask] = IMAGINED
    image[provenance != IMAGINED] = 0
    return image, provenance


def estimate_seed_color(faces: dict) -> np.ndarray:
    samples = []
    for face in faces.values():
        image = face.get("image")
        provenance = face.get("provenance")
        if image is None or provenance is None:
            continue
        mask = np.isin(provenance, [OBSERVED, RECONSTRUCTED, INFERRED])
        if mask.any():
            samples.append(image[mask].mean(axis=0))
    if not samples:
        return np.array([132, 132, 132], dtype=np.float32)
    return np.mean(np.asarray(samples), axis=0)


def build_missing_face_imaginations(faces: dict, face_reports: dict, atlas_width: int = 480) -> dict:
    """Generate missing room faces before any object-gap consideration."""
    seed_color = estimate_seed_color(faces)
    generated = {}
    for key in FACE_KEYS:
        if faces[key]["recovered"]:
            continue
        width = atlas_width
        aspect = faces[key]["heightPx"] / max(faces[key]["widthPx"], 1)
        height = max(1, min(2000, round(width * aspect)))
        base_provenance = np.full((height, width), ABSENT, dtype=np.uint8)
        structural = np.ones((height, width), dtype=bool)
        critical = np.zeros((height, width), dtype=bool)
        masks = generation_masks(base_provenance, structural, critical)
        image, provenance = deterministic_structural_imagine(
            width, height, seed_color, masks["generatable"], masks["locked"]
        )
        faces[key]["imaginedImage"] = image
        faces[key]["imaginedProvenance"] = provenance
        faces[key]["imagined"] = True
        percentages = provenance_percentages(provenance)
        generated[key] = {
            "generated": True,
            "method": "local deterministic structural Imagine fallback in canonical room-face atlas space",
            "source": "seeded from aggregate recovered structural face colour; no source-frame generation, no cloud",
            "lockedPixels": int(masks["locked"].sum()),
            "generatablePixels": int(masks["generatable"].sum()),
            "absentPixels": int(masks["absent"].sum()),
            **percentages,
        }
        face_reports[key]["imagined"] = generated[key]
    return generated


def aggregate_face_provenance(faces: dict) -> dict:
    counts = {OBSERVED: 0, RECONSTRUCTED: 0, INFERRED: 0, IMAGINED: 0, ABSENT: 0}
    total = 0
    for face in faces.values():
        provenance = face.get("provenance")
        if provenance is None:
            provenance = face.get("imaginedProvenance")
        if provenance is None:
            continue
        total += int(provenance.size)
        for code in counts:
            counts[code] += int((provenance == code).sum())
    total = max(total, 1)
    return {
        "observedPercent": counts[OBSERVED] / total * 100,
        "reconstructedPercent": counts[RECONSTRUCTED] / total * 100,
        "inferredPercent": counts[INFERRED] / total * 100,
        "imaginedPercent": counts[IMAGINED] / total * 100,
        "absentPercent": counts[ABSENT] / total * 100,
        "totalAtlasPixels": total,
    }


def measure_vram_used_mb() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=4,
        )
    except Exception:
        return None
    first = result.stdout.strip().splitlines()[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def classify_face(plane_normal_world: np.ndarray, plane_points: np.ndarray, orientation: dict) -> str:
    """Map a recovered plane onto one of the six coarse shell face keys."""
    rotation = np.asarray(orientation["rotationWorldToRoom"])
    origin = np.asarray(orientation["origin"])
    local_normal = rotation @ np.asarray(plane_normal_world)
    local_points = (plane_points - origin) @ rotation.T
    role_axis = int(np.argmax(np.abs(local_normal)))
    axis_names = {0: ("left", "right"), 1: ("floor", "ceiling"), 2: ("back", "front")}
    negative_name, positive_name = axis_names[role_axis]
    mean_offset = float(local_points[:, role_axis].mean())
    return negative_name if mean_offset < 0 else positive_name


def cluster_residual_points(points: np.ndarray, neighbor_radius_multiplier: float = 1.0, min_size: int = MIN_RESIDUAL_CLUSTER_POINTS) -> list[np.ndarray]:
    """Grid-connectivity clustering of leftover (non-planar) points.

    This is intentionally simple (no learned model, no shape prior): points
    within `radius` of one another (via a spatial hash grid, same technique
    as `export_dense_evidence.reject_isolated_points`) join one cluster.
    Each surviving cluster is reported as an axis-aligned evidence VOLUME
    (measured bounding box + mean colour), never as a completed 3D mesh.

    The radius is derived from the residual set's own expected point spacing
    (cube root of bounding volume over point count) rather than a fixed
    fraction of its bounding diagonal, so clustering scales correctly whether
    the residual set is a whole diffuse scene or an isolated compact object.
    """
    if len(points) == 0:
        return []
    extents = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    expected_spacing = float(np.prod(extents) / max(len(points), 1)) ** (1 / 3)
    radius = max(expected_spacing * neighbor_radius_multiplier, 1e-6)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index, point in enumerate(points):
        cell = tuple(np.floor(point / radius).astype(int))
        grid.setdefault(cell, []).append(index)
    visited = np.zeros(len(points), dtype=bool)
    clusters = []
    for start in range(len(points)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        member_indices = [start]
        while stack:
            current = stack.pop()
            cell = tuple(np.floor(points[current] / radius).astype(int))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for other in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                            if visited[other]:
                                continue
                            if np.linalg.norm(points[other] - points[current]) <= radius:
                                visited[other] = True
                                stack.append(other)
                                member_indices.append(other)
        if len(member_indices) >= min_size:
            clusters.append(np.asarray(member_indices))
    return clusters


def build_object_cards(residual: dict, orientation: dict, envelope: dict, max_span_fraction: float = 0.4) -> list[dict]:
    """Partial evidence-backed object placement: real measured position/size/colour, no shape completion.

    A cluster is only accepted as a candidate object if its own bounding
    span stays well below the room envelope's span on every axis
    (`max_span_fraction` of `envelope["spans"]`). Grid-connectivity
    clustering can otherwise chain together a diffuse, room-spanning
    residual cloud into one giant "object" wherever point density varies
    smoothly -- that is a clustering artifact, not furniture, and is
    rejected here rather than silently rendered as a fabricated object.
    Rejections are counted, not hidden (see `rejectedRoomSpanningClusters`
    in the returned summary from the caller).
    """
    points = np.asarray(residual["points"])
    colors = np.asarray(residual["colors"])
    supports = residual["supports"]
    rotation = np.asarray(orientation["rotationWorldToRoom"])
    origin = np.asarray(orientation["origin"])
    envelope_spans = np.asarray(envelope["spans"])
    cards = []
    rejected = 0
    accepted_index = 0
    for member_indices in cluster_residual_points(points):
        member_points = points[member_indices]
        member_colors = colors[member_indices]
        member_supports = [supports[i] for i in member_indices]
        local = (member_points - origin) @ rotation.T
        low = local.min(axis=0)
        high = local.max(axis=0)
        center = (low + high) / 2
        span = np.maximum(high - low, 1e-3)
        if np.any(span > envelope_spans * max_span_fraction):
            rejected += 1
            continue
        accepted_index += 1
        multi_view_fraction = float(np.mean([len(s) >= 2 for s in member_supports]))
        mean_color = member_colors.mean(axis=0).tolist()
        cards.append({
            "id": f"object-{accepted_index}",
            "pointCount": int(len(member_indices)),
            "localCenter": center.tolist(),
            "localSpan": span.tolist(),
            "meanColorBgr": mean_color,
            "multiViewFraction": multi_view_fraction,
            "label": "flat evidence card: measured 3D position/size/colour only, no shape completion",
        })
    return cards, rejected


def write_face_images(output: Path, key: str, image: np.ndarray | None, provenance: np.ndarray | None, prefix: str = "face") -> None:
    if image is None:
        return
    cv2.imwrite(str(output / f"{prefix}-{key}-founder.png"), image)
    palette = np.array([[24, 24, 24], [238, 238, 238], [220, 190, 120], [150, 105, 195], [90, 165, 255]], np.uint8)
    cv2.imwrite(str(output / f"{prefix}-{key}-debug.png"), palette[np.minimum(provenance, 4)])


SHELL_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Remember -- Milestone 1.0 Remember/Imagine room-shell walkthrough (private)</title>
<style>
  html,body{margin:0;height:100%;background:#0a0a0d;overflow:hidden;font-family:system-ui,sans-serif;color:#eee}
  #scene{position:absolute;inset:0;perspective:1400px;perspective-origin:50% 50%}
  #world{position:absolute;top:50%;left:50%;transform-style:preserve-3d;width:0;height:0}
  .face{position:absolute;transform-style:preserve-3d;background-size:100% 100%;
        margin-left:calc(var(--w) / -2);margin-top:calc(var(--h) / -2);
        width:var(--w);height:var(--h);opacity:0.98}
  .face.unrecovered{background:repeating-linear-gradient(45deg,rgba(255,255,255,0.05) 0 10px,rgba(255,255,255,0.0) 10px 20px);
        border:1px dashed rgba(255,120,120,0.55)}
  .face.imagined{border:1px solid rgba(90,165,255,0.75);filter:saturate(.65);opacity:.72}
  .face .label{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
        font-size:12px;color:rgba(255,170,170,0.85);text-align:center;padding:10px;pointer-events:none}
  .object{position:absolute;transform-style:preserve-3d;border:1px solid rgba(255,255,255,0.35);
        display:flex;align-items:center;justify-content:center;font-size:11px;text-align:center;
        color:#fff;background:rgba(120,170,255,0.28)}
  #hud{position:fixed;top:14px;left:14px;z-index:5;max-width:360px;font-size:12.5px;line-height:1.5}
  #hud button{background:#1c1c22;color:#eee;border:1px solid #555;border-radius:6px;padding:7px 12px;cursor:pointer;font-size:12.5px}
  #hud button:hover{background:#2a2a33}
  #metrics{position:fixed;bottom:14px;left:14px;font-size:11.5px;opacity:.8;white-space:pre-wrap;max-width:520px}
  #private-badge{position:fixed;bottom:14px;right:14px;color:#a55;font-size:11.5px;opacity:.7}
</style></head>
<body>
<div id="hud">
  <button id="modeToggle">Remember view</button>
  <button id="debugToggle">Debug provenance off</button>
  <div id="caption" style="margin-top:8px;opacity:.85">
    Drag to look around. WASD / arrow keys to move (clamped to the recovered envelope).
    Remember shows only OBSERVED/RECONSTRUCTED/INFERRED evidence. Imagine adds generic local structural
    room-face completions marked IMAGINED; ambiguous and critical unknowns remain absent. Blue boxes are
    residual-point evidence volumes (furniture-scale, position/size/colour only, no shape completion).
  </div>
</div>
<div id="metrics"></div>
<div id="private-badge">private prototype -- do not distribute source imagery</div>
<div id="scene"><div id="world" id="world"></div></div>
<script>
const DATA = __DATA_JSON__;
const world = document.getElementById('world');
let debugMode = false;
let imagineMode = false;

function makeFace(face) {
  const el = document.createElement('div');
  const visibleImagine = imagineMode && face.imagined;
  el.className = 'face' + (face.recovered ? '' : ' unrecovered') + (visibleImagine ? ' imagined' : '');
  el.style.setProperty('--w', face.widthPx + 'px');
  el.style.setProperty('--h', face.heightPx + 'px');
  el.style.transform = face.transform;
  if (face.recovered || visibleImagine || (debugMode && face.debugImage)) {
    el.style.backgroundImage = 'url(' + (debugMode ? face.debugImage : (visibleImagine ? face.imagineImage : face.founderImage)) + ')';
    el.dataset.founder = face.founderImage;
    el.dataset.imagine = face.imagineImage || face.founderImage;
    el.dataset.debug = face.debugImage;
  } else {
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = face.imagined
      ? 'ABSENT IN REMEMBER / IMAGINED STRUCTURAL FACE AVAILABLE (' + face.key + ')'
      : 'NO PLANE EVIDENCE RECOVERED (' + face.key + ')';
    el.appendChild(label);
  }
  world.appendChild(el);
}

function makeObject(card) {
  const el = document.createElement('div');
  el.className = 'object';
  el.style.width = card.widthPx + 'px';
  el.style.height = card.heightPx + 'px';
  el.style.marginLeft = (-card.widthPx / 2) + 'px';
  el.style.marginTop = (-card.heightPx / 2) + 'px';
  el.style.transform = card.transform;
  el.style.background = debugMode ? 'rgba(150,105,195,0.45)' : card.colorCss;
  el.textContent = card.id + ' (' + card.pointCount + ' pts, evidence volume)';
  world.appendChild(el);
}

function render() {
  world.innerHTML = '';
  DATA.faces.forEach(makeFace);
  DATA.objects.forEach(makeObject);
  if (debugMode) {
    document.querySelectorAll('.face[data-founder]').forEach(el => {
      el.style.backgroundImage = 'url(' + (debugMode ? el.dataset.debug : (imagineMode ? el.dataset.imagine : el.dataset.founder)) + ')';
    });
  }
}

document.getElementById('modeToggle').addEventListener('click', () => {
  imagineMode = !imagineMode;
  document.getElementById('modeToggle').textContent = imagineMode ? 'Imagine view' : 'Remember view';
  render();
});
document.getElementById('debugToggle').addEventListener('click', () => {
  debugMode = !debugMode;
  document.getElementById('debugToggle').textContent = debugMode ? 'Debug provenance on' : 'Debug provenance off';
  render();
});

document.getElementById('metrics').textContent = DATA.metricsSummary;

// -- Pointer-drag look + WASD/arrow-key walk, pure CSS 3D, no WebGL/Three.js --
let yaw = 20, pitch = -12;
let camX = 0, camY = 0, camZ = -Math.max(DATA.envelopeHalfPx.z * 2.4, 260);
const bound = DATA.envelopeHalfPx;
const keys = {};
let dragging = false, lastX = 0, lastY = 0;

document.addEventListener('keydown', e => { keys[e.key.toLowerCase()] = true; });
document.addEventListener('keyup', e => { keys[e.key.toLowerCase()] = false; });
document.getElementById('scene').addEventListener('pointerdown', e => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('pointerup', () => { dragging = false; });
window.addEventListener('pointermove', e => {
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.25;
  pitch = Math.max(-80, Math.min(80, pitch - (e.clientY - lastY) * 0.25));
  lastX = e.clientX; lastY = e.clientY;
});

function tick() {
  const speed = 4.5;
  const rad = yaw * Math.PI / 180;
  const forward = { x: Math.sin(rad) * speed, z: Math.cos(rad) * speed };
  const strafe = { x: Math.cos(rad) * speed, z: -Math.sin(rad) * speed };
  if (keys['w'] || keys['arrowup']) { camX += forward.x; camZ += forward.z; }
  if (keys['s'] || keys['arrowdown']) { camX -= forward.x; camZ -= forward.z; }
  if (keys['a'] || keys['arrowleft']) { camX -= strafe.x; camZ -= strafe.z; }
  if (keys['d'] || keys['arrowright']) { camX += strafe.x; camZ += strafe.z; }
  const margin = 40;
  camX = Math.max(-bound.x - margin, Math.min(bound.x + margin, camX));
  camZ = Math.max(-bound.z - margin, Math.min(bound.z + margin, camZ));
  camY = Math.max(-bound.y - margin, Math.min(bound.y + margin, camY));
  world.style.transform =
    'translateZ(' + (-camZ) + 'px) rotateX(' + (-pitch) + 'deg) rotateY(' + (-yaw) + 'deg) translate3d(' +
    (-camX) + 'px,' + (camY) + 'px,0px)';
  requestAnimationFrame(tick);
}
render();
requestAnimationFrame(tick);
</script>
</body></html>
"""


def write_prototype(output: Path, faces: dict, objects: list[dict], metrics: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    js_faces = []
    for key, face in faces.items():
        entry = {"key": key, "recovered": face["recovered"], "widthPx": face["widthPx"], "heightPx": face["heightPx"],
                 "transform": face["transform"]}
        if face["recovered"]:
            write_face_images(output, key, face["image"], face["provenance"])
            entry["founderImage"] = f"face-{key}-founder.png"
            entry["debugImage"] = f"face-{key}-debug.png"
        if face.get("imagined"):
            write_face_images(output, key, face["imaginedImage"], face["imaginedProvenance"], prefix="face-imagined")
            entry["imagined"] = True
            entry["imagineImage"] = f"face-imagined-{key}-founder.png"
            entry["debugImage"] = f"face-imagined-{key}-debug.png"
        else:
            entry["imagined"] = False
        js_faces.append(entry)

    js_objects = []
    for card in objects:
        color = card["meanColorBgr"]
        color_css = f"rgba({int(color[2])},{int(color[1])},{int(color[0])},0.45)"
        js_objects.append({
            "id": card["id"], "pointCount": card["pointCount"],
            "widthPx": max(20, card["localSpan"][0] * PX_PER_UNIT),
            "heightPx": max(20, card["localSpan"][1] * PX_PER_UNIT),
            "transform": (
                f"translate3d({card['localCenter'][0] * PX_PER_UNIT}px,"
                f"{-card['localCenter'][1] * PX_PER_UNIT}px,"
                f"{card['localCenter'][2] * PX_PER_UNIT}px)"
            ),
            "colorCss": color_css,
        })

    envelope_half_px = metrics["envelopeHalfPx"]
    metrics_summary = (
        f"Milestone 1.0 -- Remember faces: {metrics['shell']['facesWithEvidence']}/6 | "
        f"Imagine faces: {metrics['generation']['generatedFacesCount']} | object volumes: {len(objects)} | "
        f"orientation gravity-estimated: {metrics['orientation']['gravityEstimated']}"
    )
    data = {"faces": js_faces, "objects": js_objects, "envelopeHalfPx": envelope_half_px, "metricsSummary": metrics_summary}
    html = SHELL_TEMPLATE.replace("__DATA_JSON__", json.dumps(data))
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def assign_planes_to_faces(planes: list[dict], models: list[dict], orientation: dict, faces: dict) -> dict:
    face_evidence = {}
    for plane_meta, model in zip(planes, models):
        if plane_meta["multiViewPoints"] < MIN_PLANE_POINTS_FOR_FACE:
            continue
        key = classify_face(np.asarray(plane_meta["normal"]), model["points"], orientation)
        current_best = face_evidence.get(key)
        if current_best is None or plane_meta["multiViewPoints"] > current_best["multiViewPoints"]:
            face_evidence[key] = {"plane": plane_meta, "model": model, "multiViewPoints": plane_meta["multiViewPoints"]}

    face_reports = {}
    for key in FACE_KEYS:
        entry = face_evidence.get(key)
        if entry is None:
            faces[key]["recovered"] = False
            face_reports[key] = {"recovered": False, "reason": "no RANSAC plane matched this face within evidence threshold"}
            continue
        model = entry["model"]
        image, provenance, atlas_info = splat_atlas(model["points"], model["colors"], model["supports"])
        faces[key]["recovered"] = True
        faces[key]["image"] = image
        faces[key]["provenance"] = provenance
        face_reports[key] = {
            "recovered": True,
            "multiViewPoints": entry["multiViewPoints"],
            "distinctSourceViews": entry["plane"]["distinctSourceViews"],
            "atlas": atlas_info,
        }
    return face_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vertices = read_ply(args.ply)
    points = np.asarray([v[:3] for v in vertices], dtype=np.float64)
    colors = np.asarray([v[3:6] for v in vertices], dtype=np.uint8)
    views = parse_views(args.images_text)
    supports = read_visibility(args.visibility, len(views))
    if len(points) != len(supports):
        raise RuntimeError("PLY and visibility counts differ")
    camera_centers = np.asarray([v.center for v in views])

    orientation = canonical_orientation(points, camera_centers)
    envelope = fit_room_envelope(points, orientation)
    spans = np.asarray(envelope["spans"])
    half_extent = spans / 2 * PX_PER_UNIT

    faces = {key: {"recovered": False, "widthPx": 0, "heightPx": 0, "transform": "", "image": None, "provenance": None} for key in FACE_KEYS}
    face_dims = {
        "left": (half_extent[2] * 2, half_extent[1] * 2), "right": (half_extent[2] * 2, half_extent[1] * 2),
        "floor": (half_extent[0] * 2, half_extent[2] * 2), "ceiling": (half_extent[0] * 2, half_extent[2] * 2),
        "back": (half_extent[0] * 2, half_extent[1] * 2), "front": (half_extent[0] * 2, half_extent[1] * 2),
    }
    face_transforms = {
        "front": f"translateZ({half_extent[2]}px)",
        "back": f"rotateY(180deg) translateZ({half_extent[2]}px)",
        "right": f"rotateY(90deg) translateZ({half_extent[0]}px)",
        "left": f"rotateY(-90deg) translateZ({half_extent[0]}px)",
        "ceiling": f"rotateX(90deg) translateZ({half_extent[1]}px)",
        "floor": f"rotateX(-90deg) translateZ({half_extent[1]}px)",
    }
    for key in FACE_KEYS:
        width_px, height_px = face_dims[key]
        faces[key]["widthPx"] = max(20.0, float(width_px))
        faces[key]["heightPx"] = max(20.0, float(height_px))
        faces[key]["transform"] = face_transforms[key]

    planes, models, residual = detect_planes(args.ply, args.visibility, len(views), return_residual=True)
    generation_start = time.perf_counter()
    vram_before = measure_vram_used_mb()
    face_reports = assign_planes_to_faces(planes, models, orientation, faces)
    generated_faces = build_missing_face_imaginations(faces, face_reports)
    vram_after = measure_vram_used_mb()
    generation_latency_ms = (time.perf_counter() - generation_start) * 1000
    object_cards, rejected_clusters = build_object_cards(residual, orientation, envelope)

    faces_with_evidence = sum(1 for report in face_reports.values() if report["recovered"])
    metrics = {
        "milestone": "1.0-founder-steering",
        "orientation": orientation,
        "envelope": envelope,
        "envelopeHalfPx": {"x": float(half_extent[0]), "y": float(half_extent[1]), "z": float(half_extent[2])},
        "planesDetected": len(planes),
        "shell": {
            "faces": face_reports,
            "facesWithEvidence": faces_with_evidence,
            "facesTotal": len(FACE_KEYS),
            "shellCompletenessPercent": faces_with_evidence / len(FACE_KEYS) * 100,
            "faceProvenanceAggregate": aggregate_face_provenance(faces),
        },
        "generation": {
            "selectedApproach": "deterministic local atlas-space structural Imagine fallback",
            "whyNoDiffusionModel": "RTX 3070 CUDA is usable, but no local Diffusers install or cached generative model was available; "
                                "Azure/cloud generation is intentionally not used.",
            "pipelineSpace": "canonical room-face/atlas space, never per-source-frame generation",
            "provenancePriority": "OBSERVED > RECONSTRUCTED > INFERRED > IMAGINED > ABSENT",
            "generatedFacesCount": len(generated_faces),
            "generatedFaces": generated_faces,
            "generatedObjectGapsCount": 0,
            "generatedObjectPolicy": f"Object gap completion disabled unless confidence is adequate; cap is {MAX_GENERATED_OBJECT_GAPS}, "
                                     "and this point-only pipeline has no semantic object masks.",
            "latencyMs": generation_latency_ms,
            "vramBeforeMb": vram_before,
            "vramAfterMb": vram_after,
            "vramDeltaMb": None if vram_before is None or vram_after is None else max(0, vram_after - vram_before),
        },
        "spatialConsistency": {
            "canonicalFaceSpace": True,
            "sourceFrameGeneration": False,
            "faceTransformsPreserved": all(bool(faces[key]["transform"]) for key in FACE_KEYS),
            "generatedFacesUseExistingShellTransforms": all(
                bool(faces[key]["transform"]) for key in generated_faces
            ),
            "walkBoundsPx": {"x": float(half_extent[0]), "y": float(half_extent[1]), "z": float(half_extent[2])},
        },
        "objectPlacement": {
            "status": "partial-evidence-only",
            "method": "grid-connectivity clustering of residual (non-planar) points into axis-aligned evidence volumes; "
                      "no mesh, no shape completion, no invented appearance",
            "placedCount": len(object_cards),
            "objects": object_cards,
            "rejectedRoomSpanningClusters": rejected_clusters,
            "rejectionReason": "cluster bounding span exceeded 40% of the recovered room envelope on at least one axis "
                               "(a room-spanning clustering artifact, not plausible furniture-scale evidence)"
                               if rejected_clusters else None,
        },
    }
    if faces_with_evidence == 0:
        metrics["verdict"] = "C-FAILED"
    elif faces_with_evidence >= 3 and object_cards:
        metrics["verdict"] = "B-PARTIAL"
    else:
        metrics["verdict"] = "B-PARTIAL"

    write_prototype(args.output, faces, object_cards, metrics)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
