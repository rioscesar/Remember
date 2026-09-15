#!/usr/bin/env python3
"""Milestone 1.5: photo-anchored spatial corridor.

Replaces the generated-face "hero" cube-face walkthrough (Milestones 1.0-1.4)
with a photo-anchored corridor: two REAL captured photographs are the hero
anchors, shown at full, byte-exact fidelity and split into depth-aware
parallax layers built only from the actual photographed pixels. Between the
two heroes sits a deliberately SUBORDINATE connective shell (desaturated,
blurred, lower-resolution, reduced opacity) that is explicitly labelled as
non-photographic connective tissue, never confused with the hero evidence.

This milestone is a raw, working-demo build, not a polished/frozen
presentation (see Milestone 1.4 for that pattern): the produced HTML has no
Content-Security-Policy hardening, no product/debug mode toggle, no reset/
epoch-guard machinery, and metrics are printed directly on the page. A short
(10-20s) autoplay walkthrough and a handful of raw parallax screenshots are
produced without a browser, by directly compositing the depth layers.

Reused, unmodified building blocks (per the founder brief):
  - camera poses + spatial graph: `spatial_graph.py`'s graph.json (nodes carry
    recovered camera center/quaternion; edges carry the strong/weak evidence
    classification used to pick the two hero anchors and to decide whether
    the connective shell may reuse a rendered bridge or must stay a plain
    labelled crossfade).
  - provenance/critical masks: `evidence_doctrine.py` (OBSERVED, RECONSTRUCTED,
    INFERRED, IMAGINED, ABSENT; `generation_masks`, `bounded_inference_fill`,
    `apply_provenance_priority`, `provenance_percentages`).
  - room envelope: `milestone09_walkthrough.fit_room_envelope` /
    `canonical_orientation` (used only to report/clamp corridor extent; the
    hero photographs themselves are never reprojected into the room atlas).
  - constrained refinement: `learned_inpainting.run_learned_inpainting` at
    low strength, used ONLY on the subordinate connective shell canvas, never
    on hero pixels; fails closed to the deterministic shell exactly like
    every earlier milestone's learned engine.

Doctrine v2 guarantee tested by `milestone15_synthetic_test.py`: every
non-transparent pixel of every hero parallax layer is byte-identical to the
original captured photograph (protectedModifiedPixels == 0 by construction,
because hero pixels are never re-touched). The connective shell is always
IMAGINED and always visually subordinate to the two heroes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import (
    ABSENT,
    IMAGINED,
    INFERRED,
    OBSERVED,
    apply_provenance_priority,
    bounded_inference_fill,
    generation_masks,
    provenance_percentages,
)
from learned_inpainting import LearnedInpaintingConfig, run_learned_inpainting
from milestone09_walkthrough import canonical_orientation, fit_room_envelope
from representation_spike import Camera, quaternion_rotation

MIN_DURATION_S = 10.0
MAX_DURATION_S = 20.0
DEFAULT_DURATION_S = 15.0
DEFAULT_LAYERS = 3
SHELL_OPACITY = 0.55           # subordinate: heroes always render at opacity 1.0
SHELL_DOWNSCALE = 0.5          # subordinate: heroes always render at native resolution
SHELL_BLUR_KERNEL = 21
SHELL_SATURATION_SCALE = 0.35
# Nearest layer moves the most; farthest layer barely moves. Purely a display
# multiplier over unitless SfM coordinates (see docs/ARCHITECTURE.md, "No
# distance is displayed") -- never a measured real-world distance.
SHIFT_FACTOR_NEAR = 1.0
SHIFT_FACTOR_FAR = 0.25
SCREENSHOT_OFFSETS_PX = ((-18, 0), (0, 0), (18, 0), (0, -10))


# --------------------------------------------------------------------------
# Output safety (same pattern as milestone14_demo_freeze.require_private_output).
# --------------------------------------------------------------------------

def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_private_output(output_dir: Path, repo_root: Path) -> None:
    output = output_dir.resolve()
    git_root = repo_root.resolve()
    if output == git_root or _is_within(output, git_root):
        raise ValueError("photo-corridor raw output must remain outside the Git working tree")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Hero anchor selection over the reused spatial graph.
# --------------------------------------------------------------------------

def select_hero_pair(graph: dict, hero_a: str | None = None, hero_b: str | None = None) -> tuple[str, str, dict | None]:
    """Pick the two photo-anchor heroes from the reused spatial graph.

    Default selection: hero A is the node with the strongest total evidence
    (sum of `meanPathSupport` over its strong edges, falling back to plain
    edge degree); hero B is the other end of hero A's single best edge
    (strong preferred over weak, then highest `meanPathSupport`). Both may be
    overridden explicitly (e.g. to match a specific founder-chosen pair).
    """
    if not graph.get("edges"):
        raise ValueError("spatial graph has no edges; cannot select a connected hero pair")
    node_names = [node["name"] for node in graph["nodes"]]
    if hero_a is not None and hero_a not in node_names:
        raise ValueError(f"hero_a '{hero_a}' is not a node in the spatial graph")
    if hero_b is not None and hero_b not in node_names:
        raise ValueError(f"hero_b '{hero_b}' is not a node in the spatial graph")

    degree = Counter()
    strong_weight = Counter()
    for edge in graph["edges"]:
        degree[edge["from"]] += 1
        degree[edge["to"]] += 1
        if edge["classification"] == "strong":
            strong_weight[edge["from"]] += edge["meanPathSupport"]
            strong_weight[edge["to"]] += edge["meanPathSupport"]

    if hero_a is None:
        hero_a = max(node_names, key=lambda name: (strong_weight[name], degree[name], name))

    candidate_edges = [edge for edge in graph["edges"] if edge["from"] == hero_a or edge["to"] == hero_a]
    if not candidate_edges:
        raise ValueError(f"hero_a '{hero_a}' has no edges in the spatial graph")
    rank = {"strong": 1, "weak": 0}
    best_edge = max(candidate_edges, key=lambda edge: (rank[edge["classification"]], edge["meanPathSupport"]))

    if hero_b is None:
        hero_b = best_edge["to"] if best_edge["from"] == hero_a else best_edge["from"]
        edge = best_edge
    else:
        edge = next(
            (e for e in graph["edges"]
             if {e["from"], e["to"]} == {hero_a, hero_b}),
            None,
        )
    return hero_a, hero_b, edge


# --------------------------------------------------------------------------
# Depth-aware multiplane layering (built only from real photographed pixels).
# --------------------------------------------------------------------------

def default_camera_for_photo(width: int, height: int) -> Camera:
    """Conservative intrinsics heuristic when no cameras.txt is supplied.

    This is an APPROXIMATION (assumes a roughly normal field of view), used
    only to decide which real pixels land in which depth layer -- it never
    affects which pixels are shown, only how they are grouped for parallax.
    Reported in metrics so it is never mistaken for calibrated intrinsics.
    """
    focal = float(max(width, height))
    return Camera(width, height, focal, focal, width / 2.0, height / 2.0)


# --------------------------------------------------------------------------
# Real COLMAP intrinsics (Milestone 1.5A: real apartment capture integration).
#
# `representation_spike.parse_camera` requires the COLMAP camera model to be
# exactly PINHOLE and raises otherwise; real per-photo COLMAP exports from
# this capture pipeline use SIMPLE_RADIAL (and sometimes SIMPLE_PINHOLE),
# one distinct camera per photo. These two functions are a generalized,
# capture-agnostic reader for that case -- no photo names, paths, or
# apartment-specific constants are hardcoded here. `representation_spike`
# itself is reused unmodified (`quaternion_rotation`, the `Camera` dataclass);
# this only adds the camera-model coverage it deliberately does not provide.
# --------------------------------------------------------------------------

SUPPORTED_COLMAP_CAMERA_MODELS = {
    "PINHOLE": 4,          # fx, fy, cx, cy
    "SIMPLE_PINHOLE": 3,   # f, cx, cy
    "SIMPLE_RADIAL": 4,    # f, cx, cy, k  (radial distortion ignored below)
    "RADIAL": 5,           # f, cx, cy, k1, k2 (radial distortion ignored below)
}


def parse_colmap_cameras_txt(cameras_text: Path) -> dict[str, Camera]:
    """Parse a COLMAP `cameras.txt` into `{camera_id: Camera}`.

    Supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL and RADIAL. Radial
    distortion coefficients (if any) are not applied -- intrinsics are used
    as a pinhole approximation, exactly like `default_camera_for_photo`'s
    documented approximation, except the focal length/principal point here
    are the REAL recovered calibration values rather than a heuristic guess.
    """
    cameras: dict[str, Camera] = {}
    for line in cameras_text.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = fields[0], fields[1], int(fields[2]), int(fields[3])
        if model not in SUPPORTED_COLMAP_CAMERA_MODELS:
            raise ValueError(f"unsupported COLMAP camera model '{model}' in {cameras_text}")
        params = [float(value) for value in fields[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # SIMPLE_RADIAL, RADIAL
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        cameras[camera_id] = Camera(width, height, fx, fy, cx, cy)
    return cameras


def parse_colmap_image_camera_ids(images_text: Path) -> dict[str, str]:
    """Map `{image name: camera_id}` from a COLMAP `images.txt`.

    Only reads the pose line of each image record (every other non-comment
    line); the following 2D-point line is skipped. This is intentionally
    separate from `representation_spike.parse_views`, which discards the
    camera id -- reused here only for the id-to-name mapping it lacks.
    """
    mapping: dict[str, str] = {}
    lines = [line for line in images_text.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    for line in lines[0::2]:
        fields = line.split()
        camera_id, name = fields[8], fields[9]
        mapping[name] = camera_id
    return mapping


def load_real_camera_intrinsics(cameras_text: Path, images_text: Path) -> dict[str, Camera]:
    """Combine the two parsers above into `{photo name: Camera}`.

    Intrinsics are at the reconstruction's native calibration resolution;
    call `scale_camera_to_photo` before use if the hero photo on disk is a
    downscaled preview of that same capture.
    """
    cameras_by_id = parse_colmap_cameras_txt(cameras_text)
    camera_id_by_name = parse_colmap_image_camera_ids(images_text)
    return {
        name: cameras_by_id[camera_id]
        for name, camera_id in camera_id_by_name.items()
        if camera_id in cameras_by_id
    }


def load_pose_overrides(images_text: Path) -> dict[str, dict]:
    """`{photo name: {"center": [...], "quaternion": [...]}}` from a COLMAP
    `images.txt`, via `representation_spike.parse_views` (reused unmodified).

    The reused spatial graph (`spatial_graph.py`'s graph.json) may be built
    from a DIFFERENT reconstruction run than a given dense-evidence export
    (different photo count/coordinate frame/scale -- SfM scale is always
    arbitrary, see docs/ARCHITECTURE.md). When both exist for the same hero
    photos, the graph should still decide which two photos are heroes
    (edge/evidence classification), but the pose numbers used to project
    that reconstruction's OWN dense points must come from that SAME
    reconstruction's `images.txt`, not the graph node. This loader supplies
    that override; `run_photo_corridor`'s `pose_overrides` falls back to the
    graph node pose whenever a name is not present here.
    """
    from representation_spike import parse_views
    return {view.name: {"center": view.center, "quaternion": view.quaternion} for view in parse_views(images_text)}


def scale_camera_to_photo(camera: Camera, photo_width: int, photo_height: int) -> Camera:
    """Rescale intrinsics when the photo on disk differs from calibration size.

    Preview/downscaled photos are common (smaller than the full-resolution
    captures COLMAP was calibrated against); focal length and principal
    point both scale linearly with image size, so this is exact, not an
    approximation.
    """
    if camera.width == photo_width and camera.height == photo_height:
        return camera
    scale_x = photo_width / camera.width
    scale_y = photo_height / camera.height
    return Camera(photo_width, photo_height, camera.fx * scale_x, camera.fy * scale_y, camera.cx * scale_x, camera.cy * scale_y)


def load_critical_mask_npz(path: Path, key: str = "critical") -> np.ndarray:
    """Load a boolean critical mask from a `scene_risk_segmentation.py`-style
    `.npz` export (keys: structural/object/critical). Only the `critical`
    array is reused here -- it is never re-derived, only threaded through so
    hero depth-layer construction never treats a critical region as safe to
    bounded-fill without being counted."""
    with np.load(path) as data:
        return data[key].astype(bool)

def project_sparse_depth(
    points_xyz: np.ndarray,
    center: np.ndarray,
    quaternion: np.ndarray,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project reused 3D evidence points into this hero's photo frame.

    Returns (depth_buffer, supported_mask) at the camera's own resolution.
    Nearest point wins per pixel (occlusion-consistent with the real photo).
    """
    rotation = quaternion_rotation(np.asarray(quaternion, dtype=np.float64))
    camera_space = (points_xyz - np.asarray(center, dtype=np.float64)) @ rotation.T
    z = camera_space[:, 2]
    in_front = z > 1e-6
    u = camera.fx * camera_space[:, 0] / np.where(in_front, z, 1.0) + camera.cx
    v = camera.fy * camera_space[:, 1] / np.where(in_front, z, 1.0) + camera.cy
    px = np.round(u).astype(np.int64)
    py = np.round(v).astype(np.int64)
    in_bounds = in_front & (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)

    depth_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    if in_bounds.any():
        order = np.argsort(-z[in_bounds])  # farthest first, so nearest overwrites last.
        for x, y, depth in zip(px[in_bounds][order], py[in_bounds][order], z[in_bounds][order]):
            depth_buffer[y, x] = depth
    supported = np.isfinite(depth_buffer)
    return depth_buffer, supported


def build_depth_layers(
    photo_bgr: np.ndarray,
    points_xyz: np.ndarray,
    center: np.ndarray,
    quaternion: np.ndarray,
    camera: Camera,
    layers: int = DEFAULT_LAYERS,
    max_hole_fill_fraction: float = 0.08,
    critical_mask: np.ndarray | None = None,
) -> dict:
    """Split a hero photo into `layers` depth-ordered RGBA planes.

    Every layer's RGB is a byte-exact copy of `photo_bgr` restricted to that
    layer's mask; nothing is regenerated or reprojected. Pixels with no
    nearby 3D evidence (beyond `max_hole_fill_fraction` of the image
    diagonal) are conservatively left "flat" (shift factor 0, ABSENT depth
    provenance) rather than guessed -- they still display the real photo,
    they simply do not parallax.

    `critical_mask`, if given (a real reused critical mask, e.g. from
    `evidence_doctrine.risk_masks_from_labels`/`scene_risk_segmentation.py`,
    resized to this photo's resolution), is threaded into the bounded local
    depth fill so critical regions are never silently smoothed over when
    counting/reporting depth provenance. It never changes which real photo
    pixels are shown -- only depth-layer bookkeeping and the provenance
    screenshot use it.
    """
    height, width = photo_bgr.shape[:2]
    if layers < 2:
        raise ValueError("at least 2 depth layers are required for parallax")
    if critical_mask is None:
        critical_mask_hw = np.zeros((height, width), dtype=bool)
    else:
        critical_mask_hw = critical_mask.astype(bool)
        if critical_mask_hw.shape != (height, width):
            critical_mask_hw = cv2.resize(
                critical_mask_hw.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

    depth_buffer, supported = project_sparse_depth(points_xyz, center, quaternion, camera)
    if not supported.any():
        raise ValueError("no reused 3D evidence points projected into this hero photo; check pose/camera inputs")

    finite_depth = depth_buffer[supported]
    low, high = np.percentile(finite_depth, [1, 99])
    span = max(high - low, 1e-6)
    normalized = np.clip((depth_buffer - low) / span, 0.0, 1.0)
    depth_as_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    depth_as_bgr[supported] = np.round(normalized[supported] * 255)[:, None]

    max_distance_px = max(height, width) * max_hole_fill_fraction
    filled_bgr, delta = bounded_inference_fill(
        depth_as_bgr, supported, critical_mask=critical_mask_hw, max_distance_px=max_distance_px,
    )
    inferred = delta == INFERRED
    classified = supported | inferred
    depth_values = filled_bgr[:, :, 0].astype(np.float64)

    depth_provenance = np.full((height, width), ABSENT, dtype=np.uint8)
    depth_provenance[supported] = OBSERVED
    depth_provenance[inferred] = INFERRED

    layer_index = np.full((height, width), layers // 2, dtype=np.int16)
    edges = np.quantile(depth_values[classified], np.linspace(0.0, 1.0, layers + 1))
    edges[0] -= 1.0
    edges[-1] += 1.0
    for index in range(layers):
        band = classified & (depth_values >= edges[index]) & (depth_values < edges[index + 1])
        layer_index[band] = index

    flat_mask = ~classified  # no nearby evidence: shown, but never shifted.
    shift_by_layer = np.linspace(SHIFT_FACTOR_NEAR, SHIFT_FACTOR_FAR, layers)

    layer_records = []
    for index in range(layers):
        mask = (layer_index == index) & ~flat_mask
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[mask, :3] = photo_bgr[mask]
        rgba[mask, 3] = 255
        layer_records.append({
            "index": int(index),
            "shiftFactor": float(shift_by_layer[index]),
            "rgba": rgba,
            "pixelCount": int(mask.sum()),
        })
    # Flat/no-evidence pixels join the middle layer visually (shift 0) so the
    # full photograph always remains visible -- no holes are ever introduced.
    middle = layers // 2
    flat_rgba = layer_records[middle]["rgba"]
    flat_rgba[flat_mask, :3] = photo_bgr[flat_mask]
    flat_rgba[flat_mask, 3] = 255

    percentages = provenance_percentages(depth_provenance)
    return {
        "layers": layer_records,
        "depthProvenancePercentages": {
            "depthObservedPercent": percentages["observedPercent"],
            "depthInferredPercent": percentages["inferredPercent"],
            "depthAbsentFlatPercent": percentages["absentPercent"],
        },
        "flatPixelCount": int(flat_mask.sum()),
        "camera": {"width": camera.width, "height": camera.height, "fx": camera.fx, "fy": camera.fy},
        # Not written verbatim into metrics.json (raw arrays); consumed by
        # `render_provenance_screenshot` and by callers that report a real
        # critical-pixel count alongside depth provenance.
        "depthProvenance": depth_provenance,
        "criticalMask": critical_mask_hw if critical_mask is not None else None,
        "criticalPixelPercent": float(critical_mask_hw.mean() * 100.0) if critical_mask is not None else 0.0,
    }


def composite_offset_frame(layer_records: list[dict], width: int, height: int, offset_px: tuple[float, float]) -> np.ndarray:
    """Render one raw parallax screenshot by shifting each real-pixel layer."""
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    dx, dy = offset_px
    for record in sorted(layer_records, key=lambda r: -r["index"]):  # farthest first, nearest on top.
        shift = record["shiftFactor"]
        matrix = np.array([[1, 0, dx * shift], [0, 1, dy * shift]], dtype=np.float32)
        warped = cv2.warpAffine(
            record["rgba"], matrix, (width, height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        canvas[:, :, :3] = (warped[:, :, :3].astype(np.float32) * alpha + canvas[:, :, :3].astype(np.float32) * (1 - alpha)).astype(np.uint8)
        canvas[:, :, 3] = np.clip(canvas[:, :, 3].astype(np.float32) + warped[:, :, 3].astype(np.float32) * (1 - canvas[:, :, 3:4].astype(np.float32).squeeze(-1) / 255.0), 0, 255).astype(np.uint8)
    return canvas


# --------------------------------------------------------------------------
# Raw provenance screenshot (real evidence_doctrine.py codes, never the
# legacy scene_risk_segmentation.py provenance legend).
# --------------------------------------------------------------------------

PROVENANCE_OVERLAY_COLORS_BGR = {
    OBSERVED: (60, 200, 60),    # green: real 3D evidence supports depth here.
    INFERRED: (40, 160, 230),   # amber: bounded local fill, still real pixels.
    ABSENT: (90, 90, 90),       # gray: no nearby evidence, shown flat (no parallax).
}


def render_provenance_screenshot(
    photo_bgr: np.ndarray,
    depth_provenance: np.ndarray,
    critical_mask: np.ndarray | None = None,
    overlay_alpha: float = 0.35,
) -> np.ndarray:
    """Raw, unpolished provenance overlay for one hero photo.

    Tints the REAL photo pixels (never replaces them) by depth provenance
    (OBSERVED/INFERRED/ABSENT, `evidence_doctrine.py` codes) and outlines any
    reused critical mask in red. Produced only for reporting/inspection; the
    hero layers themselves never use this image.
    """
    overlay = photo_bgr.copy()
    for code, color in PROVENANCE_OVERLAY_COLORS_BGR.items():
        mask = depth_provenance == code
        if not mask.any():
            continue
        tint = np.zeros_like(overlay)
        tint[mask] = color
        blended = cv2.addWeighted(photo_bgr, 1 - overlay_alpha, tint, overlay_alpha, 0)
        overlay[mask] = blended[mask]
    if critical_mask is not None and critical_mask.any():
        contours, _ = cv2.findContours(critical_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (20, 20, 220), 2)
    cv2.putText(
        overlay,
        "depth: green=OBSERVED amber=INFERRED gray=ABSENT | red outline=CRITICAL (never touched)",
        (10, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return overlay


# --------------------------------------------------------------------------
# Subordinate connective shell -- never a hero, never claimed as evidence.
# --------------------------------------------------------------------------

def estimate_pair_seed_color(hero_a_bgr: np.ndarray, hero_b_bgr: np.ndarray) -> np.ndarray:
    return np.mean(
        np.stack((hero_a_bgr.reshape(-1, 3).mean(axis=0), hero_b_bgr.reshape(-1, 3).mean(axis=0))),
        axis=0,
    )


def deterministic_connective_canvas(width: int, height: int, seed_color_bgr: np.ndarray) -> np.ndarray:
    """Local deterministic connective-tissue canvas (no source photo read)."""
    y, x = np.mgrid[0:height, 0:width]
    gradient = ((x / max(width - 1, 1)) * 9 + (y / max(height - 1, 1)) * 6).astype(np.float32)
    base = np.clip(seed_color_bgr.astype(np.float32), 60, 195)
    canvas = np.clip(base + gradient[:, :, None], 0, 255).astype(np.uint8)
    return canvas


def make_subordinate(canvas_bgr: np.ndarray) -> np.ndarray:
    """Deliberately de-emphasize the shell so it never competes with a hero."""
    hsv = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= SHELL_SATURATION_SCALE
    desaturated = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(desaturated, (SHELL_BLUR_KERNEL, SHELL_BLUR_KERNEL), 0)
    small = cv2.resize(blurred, None, fx=SHELL_DOWNSCALE, fy=SHELL_DOWNSCALE, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (canvas_bgr.shape[1], canvas_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def build_connective_shell(
    hero_a_bgr: np.ndarray,
    hero_b_bgr: np.ndarray,
    width: int,
    height: int,
    engine: str = "deterministic",
    learned_config: LearnedInpaintingConfig | None = None,
    runner=None,
) -> dict:
    """Build the subordinate corridor between the two hero anchors.

    `engine`: "deterministic" (default, no model dependency), "learned"
    (fails closed to deterministic if the model/runtime is unavailable), or
    "auto" (tries learned, silently-safe fallback to deterministic).
    """
    seed_color = estimate_pair_seed_color(hero_a_bgr, hero_b_bgr)
    canvas = deterministic_connective_canvas(width, height, seed_color)
    provenance = np.full((height, width), IMAGINED, dtype=np.uint8)
    critical_mask = np.zeros((height, width), dtype=bool)  # pure synthetic canvas: no captured evidence to protect here.
    masks = generation_masks(np.full((height, width), ABSENT, dtype=np.uint8), np.ones((height, width), dtype=bool), critical_mask)

    status = {
        "requestedEngine": engine,
        "selectedEngine": "deterministic",
        "blocker": None,
    }
    if engine in {"auto", "learned"}:
        config = learned_config or LearnedInpaintingConfig(strength=0.35)
        learned = run_learned_inpainting(
            canvas, np.full((height, width), ABSENT, dtype=np.uint8), masks["generatable"], masks["locked"],
            critical_mask, face_key="corridor-shell", config=config, runner=runner,
        )
        status["learned"] = learned.metrics
        if learned.accepted and learned.image_bgr is not None and learned.provenance is not None:
            canvas, provenance = apply_provenance_priority(canvas, provenance, learned.image_bgr, learned.provenance)
            status["selectedEngine"] = "learned-diffusion"
        else:
            status["blocker"] = learned.blocker or "learned shell refinement did not validate"
            if engine == "learned":
                status["selectedEngine"] = "fail-closed-learned"

    subordinate = make_subordinate(canvas)
    return {
        "imageBgr": subordinate,
        "provenance": provenance,
        "engineStatus": status,
        "subordinate": {
            "opacity": SHELL_OPACITY,
            "renderScale": SHELL_DOWNSCALE,
            "blurKernelPx": SHELL_BLUR_KERNEL,
            "saturationScale": SHELL_SATURATION_SCALE,
            "note": "Always rendered below hero opacity/resolution; never claimed as photographic evidence.",
        },
        **provenance_percentages(provenance),
    }


# --------------------------------------------------------------------------
# Raw (unpolished, deliberately not frozen) walkthrough assembly.
# --------------------------------------------------------------------------

RAW_HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>Remember -- Milestone 1.5 raw photo corridor</title>
<style>
  body { margin: 0; background: #111; color: #ddd; font: 14px/1.4 monospace; }
  .stage { position: relative; width: 100%; height: 78vh; overflow: hidden; background: #000; }
  .stage img.layer { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; will-change: transform; }
  .shell-label { position: absolute; bottom: 8px; left: 8px; background: rgba(0,0,0,0.6); padding: 4px 8px; }
  .raw-metrics { white-space: pre-wrap; padding: 10px; border-top: 1px solid #444; }
  h1 { font-size: 15px; margin: 8px; }
</style>
<h1>Milestone 1.5 -- raw photo-anchored corridor (unpolished build)</h1>
<div class="stage" id="stage"></div>
<div class="raw-metrics" id="metrics"></div>
<script>
const DATA = __DATA_JSON__;
const stage = document.getElementById('stage');

function showHero(hero) {
  stage.innerHTML = '';
  for (const layer of hero.layers) {
    const img = document.createElement('img');
    img.className = 'layer';
    img.src = layer.file;
    img.dataset.shift = layer.shiftFactor;
    stage.appendChild(img);
  }
  document.onmousemove = (event) => {
    const dx = (event.clientX / window.innerWidth - 0.5) * 40;
    const dy = (event.clientY / window.innerHeight - 0.5) * 24;
    for (const img of stage.querySelectorAll('img.layer')) {
      const shift = parseFloat(img.dataset.shift);
      img.style.transform = 'translate(' + (dx * shift) + 'px,' + (dy * shift) + 'px)';
    }
  };
}

function showShell(shell) {
  stage.innerHTML = '';
  const img = document.createElement('img');
  img.className = 'layer';
  img.src = shell.file;
  img.style.opacity = shell.opacity;
  stage.appendChild(img);
  const label = document.createElement('div');
  label.className = 'shell-label';
  label.textContent = 'CONNECTIVE SHELL -- subordinate, not photographic evidence (IMAGINED)';
  stage.appendChild(label);
}

async function play() {
  const stops = DATA.stops;
  for (const stop of stops) {
    if (stop.kind === 'hero') showHero(stop);
    else showShell(stop);
    await new Promise((resolve) => setTimeout(resolve, stop.holdMs));
  }
}

document.getElementById('metrics').textContent = JSON.stringify(DATA.metrics, null, 2);
play();
</script>
"""


def build_walkthrough(
    output: Path,
    hero_a_name: str,
    hero_b_name: str,
    hero_a_layers: dict,
    hero_b_layers: dict,
    shell: dict,
    duration_s: float,
) -> dict:
    duration_s = min(max(duration_s, MIN_DURATION_S), MAX_DURATION_S)
    hero_hold_ms = round(duration_s * 1000 * 0.4)
    shell_hold_ms = round(duration_s * 1000 * 0.2)

    heroes_dir = output / "heroes"
    shell_dir = output / "shell"
    screenshots_dir = output / "screenshots"
    for directory in (heroes_dir, shell_dir, screenshots_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def write_hero_layers(name: str, layers: dict) -> list[dict]:
        entries = []
        for record in layers["layers"]:
            filename = f"{name}-layer-{record['index']}.png"
            cv2.imwrite(str(heroes_dir / filename), record["rgba"])
            entries.append({"file": f"heroes/{filename}", "shiftFactor": record["shiftFactor"], "index": record["index"]})
        return entries

    hero_a_entries = write_hero_layers(hero_a_name, hero_a_layers)
    hero_b_entries = write_hero_layers(hero_b_name, hero_b_layers)

    cv2.imwrite(str(shell_dir / "shell.png"), shell["imageBgr"])

    screenshot_count = 0
    hero_a_frame_paths: list[Path] = []
    hero_b_frame_paths: list[Path] = []
    for hero_name, layers, frame_paths in (
        (hero_a_name, hero_a_layers, hero_a_frame_paths),
        (hero_b_name, hero_b_layers, hero_b_frame_paths),
    ):
        height, width = layers["layers"][0]["rgba"].shape[:2]
        for offset in SCREENSHOT_OFFSETS_PX:
            frame = composite_offset_frame(layers["layers"], width, height, offset)
            path = screenshots_dir / f"{hero_name}-offset-{offset[0]}-{offset[1]}.png"
            cv2.imwrite(str(path), frame)
            frame_paths.append(path)
            screenshot_count += 1
    # Raw hard-cut recording order: hero A parallax sweep, subordinate shell,
    # hero B parallax sweep -- no crossfade, no freeze/polish.
    raw_recording_frame_paths = [*hero_a_frame_paths, shell_dir / "shell.png", *hero_b_frame_paths]

    stops = [
        {"kind": "hero", "name": hero_a_name, "layers": hero_a_entries, "holdMs": hero_hold_ms},
        {"kind": "shell", "file": "shell/shell.png", "opacity": shell["subordinate"]["opacity"], "holdMs": shell_hold_ms},
        {"kind": "hero", "name": hero_b_name, "layers": hero_b_entries, "holdMs": hero_hold_ms},
    ]
    metrics = {
        "milestone": "1.5",
        "raw": True,
        "frozen": False,
        "heroA": hero_a_name,
        "heroB": hero_b_name,
        "durationSeconds": round(duration_s, 2),
        "heroADepthProvenance": hero_a_layers["depthProvenancePercentages"],
        "heroBDepthProvenance": hero_b_layers["depthProvenancePercentages"],
        "shellProvenancePercentages": {key: shell[key] for key in (
            "observedPercent", "reconstructedPercent", "inferredPercent", "imaginedPercent", "absentPercent",
        )},
        "shellEngineStatus": shell["engineStatus"],
        "shellSubordinate": shell["subordinate"],
        "screenshotCount": screenshot_count,
    }
    (output / "index.html").write_text(
        RAW_HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps({"stops": stops, "metrics": metrics})),
        encoding="utf-8",
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    # Not written to metrics.json above (absolute local paths); attached to
    # the in-memory dict only so `run_photo_corridor` can hand the same raw
    # frames to `render_raw_recording` without re-deriving them.
    metrics["_rawRecordingFramePaths"] = raw_recording_frame_paths
    return metrics


# --------------------------------------------------------------------------
# End-to-end build.
# --------------------------------------------------------------------------

def load_photo(path: Path) -> tuple[np.ndarray, bytes]:
    raw = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode photo: {path}")
    return image, raw


def render_raw_recording(frame_paths: list[Path], output_dir: Path, duration_s: float) -> dict:
    """Concatenate already-written raw screenshot frames into one 10-20s
    recording via ffmpeg, hard cuts only (no crossfade/polish, per the raw
    Milestone 1.5A brief). Fails closed to the existing PNG frame sequence
    already on disk if ffmpeg is unavailable -- never fabricates a video.
    """
    duration_s = min(max(duration_s, MIN_DURATION_S), MAX_DURATION_S)
    ffmpeg_path = shutil.which("ffmpeg")
    hold_s = duration_s / max(len(frame_paths), 1)
    result = {
        "frameCount": len(frame_paths),
        "holdSecondsPerFrame": round(hold_s, 3),
        "expectedDurationSeconds": round(duration_s, 2),
        "style": "hard-cut, no crossfade (raw, unpolished)",
        "ffmpegAvailable": ffmpeg_path is not None,
        "produced": False,
        "outputPath": None,
        "frameSequence": [str(p) for p in frame_paths],
        "blocker": None,
    }
    if not frame_paths:
        result["blocker"] = "no raw screenshot frames available to concatenate"
        return result
    if ffmpeg_path is None:
        result["blocker"] = (
            "ffmpeg was not found on PATH locally; the reproducible PNG frame "
            "sequence above is retained as the raw recording artifact instead."
        )
        return result

    list_path = output_dir / "raw-recording-frames.txt"
    lines = []
    for frame_path in frame_paths:
        lines.append(f"file '{frame_path.resolve().as_posix()}'")
        lines.append(f"duration {hold_s:.3f}")
    lines.append(f"file '{frame_paths[-1].resolve().as_posix()}'")  # ffmpeg concat quirk: repeat last entry.
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_path = output_dir / "milestone15a-raw-walkthrough.mp4"
    cmd = [
        ffmpeg_path, "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-vf", "fps=24,format=yuv420p", str(output_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if completed.returncode == 0 and output_path.exists():
            result["produced"] = True
            result["outputPath"] = str(output_path)
        else:
            result["blocker"] = f"ffmpeg exited {completed.returncode}: {completed.stderr[-800:]}"
    except Exception as exc:  # pragma: no cover -- environment-dependent
        result["blocker"] = f"ffmpeg invocation failed: {exc}"
    return result


def run_photo_corridor(
    graph: dict,
    photos_dir: Path,
    points_xyz: np.ndarray,
    output: Path,
    hero_a: str | None = None,
    hero_b: str | None = None,
    camera: Camera | None = None,
    camera_a: Camera | None = None,
    camera_b: Camera | None = None,
    critical_mask_a: np.ndarray | None = None,
    critical_mask_b: np.ndarray | None = None,
    pose_overrides: dict[str, dict] | None = None,
    layers: int = DEFAULT_LAYERS,
    duration_s: float = DEFAULT_DURATION_S,
    shell_engine: str = "deterministic",
) -> dict:
    """Build the full raw corridor.

    `camera` applies the same intrinsics to both heroes (synthetic/legacy
    use); `camera_a`/`camera_b` take precedence when the two heroes have
    distinct real per-photo calibration (the normal real-capture case, since
    COLMAP registers one camera per photo, not one shared camera).
    `critical_mask_a`/`critical_mask_b`, if given, are real reused critical
    masks (see `load_critical_mask_npz`) resized to each hero photo; they are
    never used to alter which real pixels are shown, only to report depth
    provenance/critical-pixel bookkeeping and to draw the provenance
    screenshot outline. `pose_overrides` (see `load_pose_overrides`), if a
    hero's name is present in it, replaces that hero's graph-node pose --
    required whenever the spatial graph and `points_xyz` come from different
    reconstruction runs (the graph still decides which two photos are
    heroes; the override guarantees the pose used to project `points_xyz`
    is from that SAME run as the points, not a mismatched coordinate frame).
    """
    hero_a_name, hero_b_name, edge = select_hero_pair(graph, hero_a, hero_b)
    nodes_by_name = {node["name"]: node for node in graph["nodes"]}
    hero_a_node = nodes_by_name[hero_a_name]
    hero_b_node = nodes_by_name[hero_b_name]
    pose_overrides = pose_overrides or {}
    hero_a_pose = pose_overrides.get(hero_a_name, hero_a_node)
    hero_b_pose = pose_overrides.get(hero_b_name, hero_b_node)

    hero_a_bgr, hero_a_raw = load_photo(photos_dir / hero_a_name)
    hero_b_bgr, hero_b_raw = load_photo(photos_dir / hero_b_name)
    hero_a_hash = sha256_bytes(hero_a_raw)
    hero_b_hash = sha256_bytes(hero_b_raw)

    camera_a_final = camera_a or camera or default_camera_for_photo(hero_a_bgr.shape[1], hero_a_bgr.shape[0])
    camera_b_final = camera_b or camera or default_camera_for_photo(hero_b_bgr.shape[1], hero_b_bgr.shape[0])

    hero_a_layers = build_depth_layers(
        hero_a_bgr, points_xyz, hero_a_pose["center"], hero_a_pose["quaternion"], camera_a_final,
        layers=layers, critical_mask=critical_mask_a,
    )
    hero_b_layers = build_depth_layers(
        hero_b_bgr, points_xyz, hero_b_pose["center"], hero_b_pose["quaternion"], camera_b_final,
        layers=layers, critical_mask=critical_mask_b,
    )

    # Room-envelope orientation must use camera centers from the SAME
    # coordinate frame as `points_xyz`. When pose overrides are supplied
    # (graph and dense evidence are different reconstruction runs), only the
    # override centers are guaranteed consistent with `points_xyz` -- fall
    # back to all graph node centers only when no overrides were given.
    if pose_overrides:
        override_names = [name for name in (hero_a_name, hero_b_name) if name in pose_overrides]
        all_centers = np.stack([np.asarray(pose_overrides[name]["center"]) for name in override_names])
    else:
        all_centers = np.stack([np.asarray(node["center"]) for node in graph["nodes"]])
    orientation = canonical_orientation(points_xyz, all_centers)
    envelope = fit_room_envelope(points_xyz, orientation)

    shell_width = min(hero_a_bgr.shape[1], hero_b_bgr.shape[1])
    shell_height = min(hero_a_bgr.shape[0], hero_b_bgr.shape[0])
    shell = build_connective_shell(hero_a_bgr, hero_b_bgr, shell_width, shell_height, engine=shell_engine)

    require_private_output(output, Path(__file__).resolve().parents[1])
    output.mkdir(parents=True, exist_ok=True)
    metrics = build_walkthrough(output, hero_a_name, hero_b_name, hero_a_layers, hero_b_layers, shell, duration_s)
    raw_recording_frame_paths = metrics.pop("_rawRecordingFramePaths")

    # Raw, exactly-named artifacts requested for the real-capture rebuild:
    # byte-exact hero originals, an "entry"/"destination" screenshot at rest,
    # the subordinate mid-corridor shell, a provenance screenshot, and a
    # 10-20s raw recording (or an explicit ffmpeg-unavailable fallback).
    reference_dir = output / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    hero_a_ext = Path(hero_a_name).suffix or ".jpg"
    hero_b_ext = Path(hero_b_name).suffix or ".jpg"
    (reference_dir / f"hero-a-reference{hero_a_ext}").write_bytes(hero_a_raw)
    (reference_dir / f"hero-b-destination-original{hero_b_ext}").write_bytes(hero_b_raw)

    entry_frame = composite_offset_frame(hero_a_layers["layers"], hero_a_bgr.shape[1], hero_a_bgr.shape[0], (0, 0))
    cv2.imwrite(str(output / "entry-screenshot.png"), entry_frame)
    cv2.imwrite(str(output / "mid-corridor.png"), shell["imageBgr"])
    destination_frame = composite_offset_frame(hero_b_layers["layers"], hero_b_bgr.shape[1], hero_b_bgr.shape[0], (0, 0))
    cv2.imwrite(str(output / "hero-b-destination-screenshot.png"), destination_frame)

    provenance_dir = output / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    provenance_a = render_provenance_screenshot(hero_a_bgr, hero_a_layers["depthProvenance"], hero_a_layers["criticalMask"])
    provenance_b = render_provenance_screenshot(hero_b_bgr, hero_b_layers["depthProvenance"], hero_b_layers["criticalMask"])
    cv2.imwrite(str(provenance_dir / "provenance-hero-a.png"), provenance_a)
    cv2.imwrite(str(provenance_dir / "provenance-hero-b.png"), provenance_b)

    recording = render_raw_recording(raw_recording_frame_paths, output, duration_s)

    # Fidelity check: re-hash the exact bytes written for the hero photos'
    # nearest (fully opaque, no-hole) reference so a caller can independently
    # verify hero pixels were never modified from the original capture.
    metrics["heroFidelity"] = {
        hero_a_name: {"sha256": hero_a_hash, "widthPx": int(hero_a_bgr.shape[1]), "heightPx": int(hero_a_bgr.shape[0])},
        hero_b_name: {"sha256": hero_b_hash, "widthPx": int(hero_b_bgr.shape[1]), "heightPx": int(hero_b_bgr.shape[0])},
    }
    metrics["selectedEdge"] = edge
    metrics["roomEnvelope"] = {"spans": envelope["spans"], "method": envelope["method"]}
    metrics["criticalViolations"] = 0  # hero pixels are never regenerated; shell is a pure synthetic canvas.
    metrics["heroACriticalPixelPercent"] = hero_a_layers["criticalPixelPercent"]
    metrics["heroBCriticalPixelPercent"] = hero_b_layers["criticalPixelPercent"]
    metrics["rawRecording"] = recording
    metrics["rawArtifacts"] = {
        "heroAReference": str(reference_dir / f"hero-a-reference{hero_a_ext}"),
        "entryScreenshot": str(output / "entry-screenshot.png"),
        "midCorridor": str(output / "mid-corridor.png"),
        "heroBDestinationOriginal": str(reference_dir / f"hero-b-destination-original{hero_b_ext}"),
        "heroBDestinationScreenshot": str(output / "hero-b-destination-screenshot.png"),
        "provenanceScreenshotHeroA": str(provenance_dir / "provenance-hero-a.png"),
        "provenanceScreenshotHeroB": str(provenance_dir / "provenance-hero-b.png"),
        "rawRecording": recording.get("outputPath"),
        "metrics": str(output / "metrics.json"),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True, help="spatial_graph.py graph.json (private)")
    parser.add_argument("--photos-dir", type=Path, required=True, help="directory containing the source photographs")
    parser.add_argument("--dense-evidence", type=Path, required=True, help="export_dense_evidence.py output JSON")
    parser.add_argument("--output", type=Path, required=True, help="private raw output directory (must stay outside Git)")
    parser.add_argument("--hero-a", type=str, default=None)
    parser.add_argument("--hero-b", type=str, default=None)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--shell-engine", choices=["deterministic", "learned", "auto"], default="deterministic")
    parser.add_argument("--camera-text", type=Path, default=None, help="optional COLMAP PINHOLE cameras.txt (single shared camera)")
    parser.add_argument(
        "--cameras-text", type=Path, default=None,
        help="optional COLMAP cameras.txt (PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL, one camera per photo)",
    )
    parser.add_argument(
        "--images-text", type=Path, default=None,
        help="optional COLMAP images.txt, required alongside --cameras-text to map photo name to camera id",
    )
    parser.add_argument("--critical-mask-a", type=Path, default=None, help="optional scene_risk_segmentation.py .npz for hero A")
    parser.add_argument("--critical-mask-b", type=Path, default=None, help="optional scene_risk_segmentation.py .npz for hero B")
    parser.add_argument(
        "--poses-text", type=Path, default=None,
        help="optional COLMAP images.txt from the SAME reconstruction run as --dense-evidence, used to override a "
             "hero's pose when the spatial graph was built from a different reconstruction run",
    )
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    dense = json.loads(args.dense_evidence.read_text(encoding="utf-8"))
    points_xyz = np.array([[point["x"], point["y"], point["z"]] for point in dense["points"]], dtype=np.float64)

    camera = None
    if args.camera_text is not None:
        from representation_spike import parse_camera
        camera = parse_camera(args.camera_text)

    camera_a = camera_b = None
    if args.cameras_text is not None and args.images_text is not None:
        hero_a_name, hero_b_name, _ = select_hero_pair(graph, args.hero_a, args.hero_b)
        real_intrinsics = load_real_camera_intrinsics(args.cameras_text, args.images_text)
        photo_sizes = {}
        for name in (hero_a_name, hero_b_name):
            photo, _ = load_photo(args.photos_dir / name)
            photo_sizes[name] = (photo.shape[1], photo.shape[0])
        if hero_a_name in real_intrinsics:
            width, height = photo_sizes[hero_a_name]
            camera_a = scale_camera_to_photo(real_intrinsics[hero_a_name], width, height)
        if hero_b_name in real_intrinsics:
            width, height = photo_sizes[hero_b_name]
            camera_b = scale_camera_to_photo(real_intrinsics[hero_b_name], width, height)

    critical_mask_a = load_critical_mask_npz(args.critical_mask_a) if args.critical_mask_a is not None else None
    critical_mask_b = load_critical_mask_npz(args.critical_mask_b) if args.critical_mask_b is not None else None

    pose_overrides = load_pose_overrides(args.poses_text) if args.poses_text is not None else None

    started = time.perf_counter()
    metrics = run_photo_corridor(
        graph, args.photos_dir, points_xyz, args.output,
        hero_a=args.hero_a, hero_b=args.hero_b, camera=camera, camera_a=camera_a, camera_b=camera_b,
        critical_mask_a=critical_mask_a, critical_mask_b=critical_mask_b, pose_overrides=pose_overrides,
        layers=args.layers, duration_s=args.duration_seconds, shell_engine=args.shell_engine,
    )
    metrics["buildLatencyMs"] = round((time.perf_counter() - started) * 1000, 1)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote raw photo corridor to {args.output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
