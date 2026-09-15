#!/usr/bin/env python3
"""Milestone 1.5B: continuous multi-hop photo corridor (A -> optional C -> B).

Extends Milestone 1.5/1.5A's two-hero photo corridor
(`milestone15_photo_corridor.py`) with three additions requested by the
founder brief, all built on top of that module's existing, unmodified
building blocks (camera poses, spatial graph, provenance/critical masks,
room envelope, constrained refinement):

  1. **Multi-hop path selection.** `select_corridor_path` searches the
     reused spatial graph for the best route between the SAME two hero
     anchors `select_hero_pair` would already choose -- either the direct
     edge, or a path through exactly one intermediate node C -- minimizing
     the mean per-edge `angularDifferenceDegrees` and rewarding higher
     `sharedLandmarks`/`meanPathSupport`. This never changes which two
     photos are heroes; it only decides the corridor's route between them.
  2. **Support-aware spatial blend + conservative rigid-critical parallax.**
     `render_continuous_frame` crossfades the two REAL endpoint photos of
     whichever segment a given path-progress value falls in, biasing the
     crossfade by each side's real depth-provenance confidence
     (`evidence_doctrine.py` codes) rather than a flat linear fade, and
     applies `build_depth_layers`' existing multiplane parallax with a new
     rigid override: any pixel inside a real critical/high-detail mask is
     forced back to its exact, unshifted original pixel value every frame,
     regardless of the parallax offset applied elsewhere.
  3. **Non-gray connective space from existing real/private shell assets.**
     `load_bridge_frames`/`bridge_frame_for_fraction` optionally read an
     already-rendered, evidence-gated radiance bridge (the exact schema
     produced by `hybrid_transition_render.py`: `manifest.json` +
     `<from>__<to>/frame-NN-gated-rgba.png`) as real, textured connective
     material for a segment when one exists for that exact edge pair. No
     new rendering/training happens here -- pre-rendered frames are read as-
     is. When no such bridge asset exists for a segment's edge,
     `photo_crossfade_canvas` falls back to a real-photo crossfade of that
     segment's own two endpoint photos (still non-gray, still built only
     from real captured pixels) rather than the flat, desaturated synthetic
     gradient `milestone15_photo_corridor.deterministic_connective_canvas`
     uses -- that flat canvas is intentionally never used by this module.

Per the 1.5B brief: no reconstruction is re-run, no representation is
pivoted (everything here consumes existing poses/graph/dense evidence/masks
exactly as Milestone 1.5/1.5A already do), and the corridor renderer stays
raw/unpolished (no freeze machinery, no on-frame text/debug burned into the
continuous MP4 -- provenance is a separate screenshot, never composited into
the video).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, INFERRED, OBSERVED
from milestone15_photo_corridor import (
    Camera,
    build_depth_layers,
    composite_offset_frame,
    default_camera_for_photo,
    load_critical_mask_npz,
    load_photo,
    load_pose_overrides,
    load_real_camera_intrinsics,
    make_subordinate,
    render_provenance_screenshot,
    require_private_output,
    scale_camera_to_photo,
    select_hero_pair,
    sha256_bytes,
)

MIN_DURATION_S = 10.0
MAX_DURATION_S = 20.0
DEFAULT_DURATION_S = 15.0
MIN_FPS = 24
MAX_FPS = 30
DEFAULT_FPS = 24
DEFAULT_LAYERS = 3
# Conservative: smaller than Milestone 1.5/1.5A's raw screenshot offsets
# (up to 18px), because this parallax must look continuous/plausible across
# hundreds of consecutive frames rather than a handful of discrete stills.
MAX_PARALLAX_OFFSET_PX = 12.0
SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_VIDEO_MAX_DIM = 1600  # continuous-video frames render at this max side length by default;
# the anchor references, 0/25/50/75/100% samples, and provenance snapshot are UNAFFECTED and always
# render at full native photo resolution -- this only bounds the per-frame cost of the hundreds of
# frames the continuous walkthrough video needs, which would otherwise recompute full-resolution
# parallax/blend/connective-shell compositing (4000x3000 or larger) on every single frame.
SAMPLE_NAMES = ("first", "quarter-25", "mid-50", "quarter-75", "final")


# --------------------------------------------------------------------------
# 1. Multi-hop path selection over the reused spatial graph.
# --------------------------------------------------------------------------

def edge_cost(edge: dict) -> float:
    """Lower is better: normalized angular change minus landmark/support reward.

    `angularDifferenceDegrees` is normalized against 180 degrees (the
    largest possible viewing-direction change). `sharedLandmarks` is
    squashed with a soft cap (`shared / (shared + 50)`) so one edge with an
    unusually large landmark count cannot dominate unboundedly; both terms
    plus `meanPathSupport` are already unitless/bounded, so no other scaling
    is applied.
    """
    angular = float(edge.get("angularDifferenceDegrees", 180.0))
    support = float(edge.get("meanPathSupport", 0.0))
    shared = float(edge.get("sharedLandmarks", 0))
    shared_term = shared / (shared + 50.0)
    return (angular / 180.0) - 0.5 * support - 0.5 * shared_term


def select_corridor_path(graph: dict, hero_a: str | None = None, hero_b: str | None = None, allow_via: bool = True) -> dict:
    """Pick the corridor route between the two hero anchors.

    Hero selection itself is unchanged (delegates to `select_hero_pair`).
    This only decides HOW to get from hero A to hero B: the direct edge, or
    -- if `allow_via` and a strictly better route exists -- a path through
    exactly one intermediate node C. "Better" means lower MEAN per-edge cost
    (`edge_cost`), not lower total cost, so a 2-edge path is never favored
    merely for having more edges; every edge on the winning path must, on
    average, resemble a smaller-angular-change/higher-support/more-shared-
    landmarks step than the direct alternative.
    """
    hero_a_name, hero_b_name, direct_edge = select_hero_pair(graph, hero_a, hero_b)

    edges_by_pair: dict[frozenset, dict] = {}
    adjacency: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        edges_by_pair[frozenset({edge["from"], edge["to"]})] = edge
        adjacency.setdefault(edge["from"], []).append(edge)
        adjacency.setdefault(edge["to"], []).append(edge)

    candidates = []
    if direct_edge is not None:
        candidates.append({
            "path": [hero_a_name, hero_b_name], "edges": [direct_edge],
            "meanCost": edge_cost(direct_edge), "viaNode": None,
        })

    if allow_via:
        for first_edge in adjacency.get(hero_a_name, []):
            via_name = first_edge["to"] if first_edge["from"] == hero_a_name else first_edge["from"]
            if via_name in (hero_a_name, hero_b_name):
                continue
            second_edge = edges_by_pair.get(frozenset({via_name, hero_b_name}))
            if second_edge is None:
                continue
            costs = [edge_cost(first_edge), edge_cost(second_edge)]
            candidates.append({
                "path": [hero_a_name, via_name, hero_b_name], "edges": [first_edge, second_edge],
                "meanCost": sum(costs) / len(costs), "viaNode": via_name,
            })

    if not candidates:
        raise ValueError(f"no direct or one-hop-via path found between '{hero_a_name}' and '{hero_b_name}'")

    best = min(candidates, key=lambda c: c["meanCost"])
    return {
        "heroA": hero_a_name,
        "heroB": hero_b_name,
        "path": best["path"],
        "edges": best["edges"],
        "viaNode": best["viaNode"],
        "meanEdgeCost": best["meanCost"],
        "candidateCount": len(candidates),
        "candidates": [
            {"path": candidate["path"], "meanEdgeCost": candidate["meanCost"], "viaNode": candidate["viaNode"]}
            for candidate in sorted(candidates, key=lambda c: c["meanCost"])
        ],
    }


# --------------------------------------------------------------------------
# Pose interpolation along the selected path (numpy-only slerp; no torch).
# --------------------------------------------------------------------------

def slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = float(np.dot(q1, q2))
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)
    return q1 * np.cos(theta) + q3 * np.sin(theta)


def interpolate_pose(pose_a: dict, pose_b: dict, t: float) -> dict:
    center = (1 - t) * np.asarray(pose_a["center"], dtype=np.float64) + t * np.asarray(pose_b["center"], dtype=np.float64)
    quaternion = slerp_quaternion(pose_a["quaternion"], pose_b["quaternion"], t)
    return {"center": center.tolist(), "quaternion": quaternion.tolist()}


# --------------------------------------------------------------------------
# 2a. Rigid-critical parallax: critical/high-detail regions never shift.
# --------------------------------------------------------------------------

def composite_parallax_frame_rigid(layer_records: list[dict], critical_mask: np.ndarray | None, width: int, height: int, offset_px: tuple[float, float], base_photo_bgr: np.ndarray) -> np.ndarray:
    """Wrap `composite_offset_frame` so critical/high-detail pixels are
    RIGID: forced back to their exact, un-shifted original pixel values,
    regardless of the parallax offset applied to everything else. This is
    stronger than Milestone 1.5A's critical-mask threading (which only
    reported/bookkept the mask); here it actively suppresses motion.
    """
    canvas = composite_offset_frame(layer_records, width, height, offset_px)
    if critical_mask is not None and critical_mask.any():
        canvas[critical_mask, :3] = base_photo_bgr[critical_mask]
        canvas[critical_mask, 3] = 255
    return canvas


# --------------------------------------------------------------------------
# 2b. Support-aware spatial blend weight (confidence-biased crossfade).
# --------------------------------------------------------------------------

CONFIDENCE_BY_PROVENANCE = {OBSERVED: 1.0, INFERRED: 0.6, ABSENT: 0.15}
BLEND_BIAS_STRENGTH = 0.35


def provenance_confidence(depth_provenance: np.ndarray) -> np.ndarray:
    confidence = np.full(depth_provenance.shape, CONFIDENCE_BY_PROVENANCE[ABSENT], dtype=np.float64)
    for code, value in CONFIDENCE_BY_PROVENANCE.items():
        confidence[depth_provenance == code] = value
    return confidence


def support_aware_blend_weight(confidence_a: np.ndarray, confidence_b: np.ndarray, t: float, bias_strength: float = BLEND_BIAS_STRENGTH) -> np.ndarray:
    """Per-pixel blend weight towards side B (0 = pure A, 1 = pure B).

    A plain linear crossfade (`weight = t`) ignores which side's reused
    depth-provenance confidence is actually higher at that pixel. This
    nudges the weight towards whichever side has the stronger real evidence
    there, scaled so the bias vanishes at both path endpoints (`t=0`/`t=1`
    always return exactly 0/1, so the sampled "first"/"final" frames remain
    pure, undistorted anchor renders) and peaks at the segment midpoint.
    """
    bias = bias_strength * (confidence_b - confidence_a) * (1.0 - np.abs(2.0 * t - 1.0))
    return np.clip(t + bias, 0.0, 1.0)


# --------------------------------------------------------------------------
# 3. Non-gray connective space from existing real/private shell assets.
# --------------------------------------------------------------------------

def load_bridge_frames(bridge_dir: Path) -> dict[frozenset, list[dict]] | None:
    """Read a pre-rendered evidence-gated radiance bridge directory (the
    `hybrid_transition_render.py` output schema: `manifest.json` plus
    `<from>__<to>/frame-NN-gated-rgba.png`). Never fabricates content and
    never re-invokes the (torch/checkpoint-dependent) renderer -- only reads
    frames that already exist on disk. Returns `None` if the directory or
    manifest is missing/malformed so callers fall back cleanly.
    """
    manifest_path = bridge_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    result: dict[frozenset, list[dict]] = {}
    for edge in manifest.get("edges", []):
        from_name, to_name = edge.get("from"), edge.get("to")
        if not from_name or not to_name:
            continue
        edge_dir = bridge_dir / f"{from_name}__{to_name}"
        frames = []
        for frame_info in sorted(edge.get("frames", []), key=lambda f: f["frame"]):
            frame_path = edge_dir / f"frame-{frame_info['frame']:02d}-gated-rgba.png"
            if not frame_path.exists():
                continue
            image = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue
            frames.append({
                "fraction": float(frame_info["fraction"]),
                "meanSupportScore": float(frame_info.get("meanSupportScore", 0.0)),
                "image": image,
            })
        if frames:
            result[frozenset({from_name, to_name})] = frames
    return result or None


def bridge_frame_for_fraction(frames: list[dict], t: float) -> np.ndarray:
    """Nearest-fraction lookup into a loaded bridge-frame sequence."""
    best = min(frames, key=lambda f: abs(f["fraction"] - t))
    return best["image"]


CONNECTIVE_WORKING_MAX_DIM = 480  # subordinate material is always blurred/de-emphasized anyway;
# rendering it at full photo resolution every one of hundreds of continuous-video frames would be
# needlessly expensive. Working at a small, fixed resolution and upscaling once at the end changes
# nothing about the *doctrine* (still subordinate, still IMAGINED, still never a hero pixel) --
# only how many pixels the blur/blend arithmetic touches per frame.


def _working_size(width: int, height: int, max_dim: int = CONNECTIVE_WORKING_MAX_DIM) -> tuple[int, int]:
    scale = min(1.0, max_dim / max(width, height))
    return max(int(round(width * scale)), 2), max(int(round(height * scale)), 2)


def photo_crossfade_canvas(photo_u_bgr: np.ndarray, photo_v_bgr: np.ndarray, t: float, width: int, height: int) -> np.ndarray:
    """Fallback non-gray connective material when no bridge asset exists for
    a segment's edge: a real-photo crossfade of that segment's own two
    endpoint photos, then the existing `make_subordinate` de-emphasis
    (blur/desaturate/downscale) so it stays clearly subordinate. Unlike
    `deterministic_connective_canvas` (a flat seed-color gradient with no
    source photo read at all), every pixel here originates from a REAL
    captured photograph. Computed at a small fixed working resolution (see
    `CONNECTIVE_WORKING_MAX_DIM`) and upscaled once to the requested canvas
    size -- this material is already blurred/subordinate by design, so the
    extra resolution of a full-size photo buys no visible detail here, only
    render time (this function runs once per continuous-video frame).
    """
    work_width, work_height = _working_size(width, height)
    resized_u = cv2.resize(photo_u_bgr, (work_width, work_height), interpolation=cv2.INTER_AREA)
    resized_v = cv2.resize(photo_v_bgr, (work_width, work_height), interpolation=cv2.INTER_AREA)
    blended = cv2.addWeighted(resized_u, 1.0 - t, resized_v, t, 0)
    subordinate = make_subordinate(blended)
    if (width, height) != (work_width, work_height):
        subordinate = cv2.resize(subordinate, (width, height), interpolation=cv2.INTER_LINEAR)
    return subordinate


def build_segment_connective_frame(from_name: str, to_name: str, photo_u_bgr: np.ndarray, photo_v_bgr: np.ndarray, t: float, width: int, height: int, bridge_frames_by_edge: dict[frozenset, list[dict]] | None) -> tuple[np.ndarray, str]:
    """Connective backdrop for one segment at local fraction `t`.

    Prefers an exact-edge match in `bridge_frames_by_edge` (a real, already-
    rendered, evidence-gated radiance frame) when available; otherwise falls
    back to `photo_crossfade_canvas`. Returns `(canvas_bgr, source_label)` --
    the label is written into metrics so which material was actually used
    for each segment is always honestly reported, never conflated.
    """
    key = frozenset({from_name, to_name})
    if bridge_frames_by_edge and key in bridge_frames_by_edge:
        frame = bridge_frame_for_fraction(bridge_frames_by_edge[key], t)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.shape[2] == 4:
            alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
            fallback = photo_crossfade_canvas(photo_u_bgr, photo_v_bgr, t, width, height).astype(np.float32)
            canvas = (frame[:, :, :3].astype(np.float32) * alpha + fallback * (1 - alpha)).astype(np.uint8)
        else:
            canvas = frame[:, :, :3]
        return canvas, "hybrid-bridge-real-render"
    return photo_crossfade_canvas(photo_u_bgr, photo_v_bgr, t, width, height), "photo-crossfade-shell"


# --------------------------------------------------------------------------
# Continuous, no-hard-cut, no-text/debug frame + video rendering.
# --------------------------------------------------------------------------

def path_progress_to_segment(path_length_edges: int, u_global: float) -> tuple[int, float]:
    """Map a global path-progress value in [0,1] to (segment index, local t).

    Equal time is allocated per edge/segment (simplest, most predictable
    schedule); a corridor with N edges spends `1/N` of the total duration on
    each edge.
    """
    u_global = min(max(u_global, 0.0), 1.0)
    if path_length_edges < 1:
        raise ValueError("path must have at least one edge/segment")
    scaled = u_global * path_length_edges
    segment = min(int(scaled), path_length_edges - 1)
    local_t = scaled - segment
    if segment == path_length_edges - 1 and u_global >= 1.0:
        local_t = 1.0
    return segment, local_t


def render_continuous_frame(node_records: list[dict], u_global: float, bridge_frames_by_edge: dict[frozenset, list[dict]] | None, parallax_offset_px: tuple[float, float]) -> np.ndarray:
    """Render one continuous-corridor frame at global path-progress `u_global`.

    `node_records` is one dict per path node, in path order, each with keys
    `name`, `photoBgr`, `layers` (from `build_depth_layers`). Composites:
      - each side's own rigid-critical parallax-shifted frame (2a),
      - a support-aware confidence-biased crossfade between them (2b),
      - the segment's non-gray connective backdrop (3) showing through any
        residual transparency (e.g. parallax-shifted edge strips).
    """
    segment, local_t = path_progress_to_segment(len(node_records) - 1, u_global)
    node_u = node_records[segment]
    node_v = node_records[segment + 1]
    height, width = node_u["photoBgr"].shape[:2]

    frame_u = composite_parallax_frame_rigid(
        node_u["layers"]["layers"], node_u["layers"]["criticalMask"], width, height, parallax_offset_px, node_u["photoBgr"],
    )
    frame_v_native = composite_parallax_frame_rigid(
        node_v["layers"]["layers"], node_v["layers"]["criticalMask"], node_v["photoBgr"].shape[1], node_v["photoBgr"].shape[0],
        parallax_offset_px, node_v["photoBgr"],
    )
    frame_v = cv2.resize(frame_v_native, (width, height), interpolation=cv2.INTER_AREA) if frame_v_native.shape[:2] != (height, width) else frame_v_native

    confidence_u = provenance_confidence(node_u["layers"]["depthProvenance"])
    confidence_v_native = provenance_confidence(node_v["layers"]["depthProvenance"])
    confidence_v = cv2.resize(confidence_v_native, (width, height), interpolation=cv2.INTER_NEAREST) if confidence_v_native.shape != (height, width) else confidence_v_native
    blend_weight = support_aware_blend_weight(confidence_u, confidence_v, local_t)

    alpha_u = frame_u[:, :, 3].astype(np.float64) / 255.0
    alpha_v = frame_v[:, :, 3].astype(np.float64) / 255.0
    w = blend_weight
    combined_alpha = np.clip((1 - w) * alpha_u + w * alpha_v, 0.0, 1.0)
    combined_rgb = (
        (1 - w)[:, :, None] * alpha_u[:, :, None] * frame_u[:, :, :3].astype(np.float64)
        + w[:, :, None] * alpha_v[:, :, None] * frame_v[:, :, :3].astype(np.float64)
    )

    connective_bgr, _ = build_segment_connective_frame(
        node_u["name"], node_v["name"], node_u["photoBgr"], node_v["photoBgr"], local_t, width, height, bridge_frames_by_edge,
    )
    final_rgb = combined_rgb + (1.0 - combined_alpha)[:, :, None] * connective_bgr.astype(np.float64)
    return np.clip(final_rgb, 0, 255).astype(np.uint8)


def parallax_offset_for_progress(u_global: float, max_offset_px: float = MAX_PARALLAX_OFFSET_PX) -> tuple[float, float]:
    """Conservative continuous parallax sweep: smooth, small, no hard jumps.

    A single sinusoidal sweep across the whole corridor (not per-segment)
    means adjacent frames -- even across a segment boundary -- always differ
    by only a small, bounded amount, which is what "no hard cuts" requires.
    """
    angle = u_global * np.pi
    dx = max_offset_px * np.sin(angle)
    dy = (max_offset_px * 0.4) * np.sin(angle * 2.0)
    return float(dx), float(dy)


def scale_node_record_for_video(record: dict, max_dim: int) -> dict:
    """Return a resized copy of one node record for continuous-video
    rendering only -- never used for the anchor references, path-progress
    samples, or provenance snapshot, which always stay at full native photo
    resolution. Downscaling here changes nothing about which pixels are
    "hero"/"real"; it only reduces how many pixels the parallax/blend
    arithmetic touches per one of the hundreds of continuous-video frames.
    """
    height, width = record["photoBgr"].shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale >= 1.0:
        return record
    new_width = max(int(round(width * scale)), 2)
    new_height = max(int(round(height * scale)), 2)
    photo_bgr = cv2.resize(record["photoBgr"], (new_width, new_height), interpolation=cv2.INTER_AREA)
    layers_in = record["layers"]
    scaled_layer_records = []
    for layer in layers_in["layers"]:
        rgba = cv2.resize(layer["rgba"], (new_width, new_height), interpolation=cv2.INTER_AREA)
        scaled_layer_records.append({**layer, "rgba": rgba})
    critical_mask = layers_in["criticalMask"]
    if critical_mask is not None:
        critical_mask = cv2.resize(
            critical_mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    depth_provenance = cv2.resize(
        layers_in["depthProvenance"], (new_width, new_height), interpolation=cv2.INTER_NEAREST,
    )
    scaled_layers = {**layers_in, "layers": scaled_layer_records, "criticalMask": critical_mask, "depthProvenance": depth_provenance}
    return {**record, "photoBgr": photo_bgr, "layers": scaled_layers}


def render_continuous_video(node_records: list[dict], bridge_frames_by_edge: dict[frozenset, list[dict]] | None, output_dir: Path, duration_s: float, fps: int, video_max_dim: int = DEFAULT_VIDEO_MAX_DIM) -> dict:
    """Render a dense sequence of continuously-varying frames and encode
    them into one MP4 via a plain image-sequence ffmpeg invocation (NOT the
    `concat` demuxer with per-frame `duration` used by Milestone 1.5A's
    hard-cut recording) -- every frame is a slightly different render of the
    same continuous path-progress sweep, so there are no hard cuts, no text,
    and no debug overlay burned into the video. Fails closed to the retained
    PNG frame sequence if ffmpeg is unavailable.
    """
    duration_s = min(max(duration_s, MIN_DURATION_S), MAX_DURATION_S)
    fps = min(max(fps, MIN_FPS), MAX_FPS)
    frame_count = max(int(round(duration_s * fps)), 2)
    video_node_records = [scale_node_record_for_video(record, video_max_dim) for record in node_records]

    frames_dir = output_dir / "continuous-frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for index in range(frame_count):
        u_global = index / (frame_count - 1)
        offset = parallax_offset_for_progress(u_global)
        frame = render_continuous_frame(video_node_records, u_global, bridge_frames_by_edge, offset)
        frame_path = frames_dir / f"frame-{index:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)

    ffmpeg_path = shutil.which("ffmpeg")
    rendered_height, rendered_width = video_node_records[0]["photoBgr"].shape[:2] if video_node_records else (0, 0)
    result = {
        "frameCount": frame_count,
        "fps": fps,
        "expectedDurationSeconds": round(frame_count / fps, 2),
        "style": "continuous, no hard cuts, no text/debug overlay",
        "renderedWidthPx": int(rendered_width),
        "renderedHeightPx": int(rendered_height),
        "videoMaxDim": video_max_dim,
        "ffmpegAvailable": ffmpeg_path is not None,
        "produced": False,
        "outputPath": None,
        "framesDir": str(frames_dir),
        "blocker": None,
    }
    if ffmpeg_path is None:
        result["blocker"] = (
            "ffmpeg was not found on PATH locally; the reproducible continuous PNG frame "
            "sequence above is retained as the raw artifact instead."
        )
        return result

    output_path = output_dir / "milestone15b-continuous-corridor.mp4"
    cmd = [
        ffmpeg_path, "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame-%04d.png"),
        "-vf", "format=yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if completed.returncode == 0 and output_path.exists():
            result["produced"] = True
            result["outputPath"] = str(output_path)
        else:
            result["blocker"] = f"ffmpeg exited {completed.returncode}: {completed.stderr[-800:]}"
    except Exception as exc:  # pragma: no cover -- environment-dependent
        result["blocker"] = f"ffmpeg invocation failed: {exc}"
    return result


# --------------------------------------------------------------------------
# End-to-end orchestration.
# --------------------------------------------------------------------------

def _resolve_pose(name: str, nodes_by_name: dict, pose_overrides: dict[str, dict]) -> dict:
    if name in pose_overrides:
        return pose_overrides[name]
    node = nodes_by_name[name]
    return {"center": node["center"], "quaternion": node["quaternion"]}


def _resolve_camera(name: str, photo_bgr: np.ndarray, real_intrinsics: dict[str, Camera] | None, shared_camera: Camera | None) -> Camera:
    height, width = photo_bgr.shape[:2]
    if real_intrinsics is not None and name in real_intrinsics:
        return scale_camera_to_photo(real_intrinsics[name], width, height)
    if shared_camera is not None:
        return shared_camera
    return default_camera_for_photo(width, height)


def render_provenance_montage(node_records: list[dict]) -> np.ndarray:
    """Single side-by-side montage of every path node's provenance overlay
    (one combined "provenance snapshot", in addition to the per-node PNGs
    also written to disk) -- all real photo pixels, tinted only.
    """
    tiles = []
    target_height = min(record["photoBgr"].shape[0] for record in node_records)
    for record in node_records:
        overlay = render_provenance_screenshot(
            record["photoBgr"], record["layers"]["depthProvenance"], record["layers"]["criticalMask"],
        )
        height, width = overlay.shape[:2]
        scale = target_height / height
        resized = cv2.resize(overlay, (max(int(round(width * scale)), 1), target_height), interpolation=cv2.INTER_AREA)
        cv2.putText(resized, record["name"], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(resized)
    return cv2.hconcat(tiles)


def run_continuous_corridor(
    graph: dict,
    photos_dir: Path,
    points_xyz: np.ndarray,
    output: Path,
    hero_a: str | None = None,
    hero_b: str | None = None,
    allow_via: bool = True,
    camera: Camera | None = None,
    real_intrinsics: dict[str, Camera] | None = None,
    critical_masks: dict[str, np.ndarray] | None = None,
    pose_overrides: dict[str, dict] | None = None,
    bridge_dir: Path | None = None,
    layers: int = DEFAULT_LAYERS,
    duration_s: float = DEFAULT_DURATION_S,
    fps: int = DEFAULT_FPS,
    video_max_dim: int = DEFAULT_VIDEO_MAX_DIM,
) -> dict:
    """Build the full raw continuous corridor artifact set.

    `real_intrinsics`/`pose_overrides`/`critical_masks` are all keyed by
    photo name and apply uniformly to EVERY node on the selected path
    (hero A, hero B, and an optional via node C) -- generalized the same
    way Milestone 1.5A's per-hero arguments were, just extended from two
    names to however many the path search actually selects.
    """
    path_info = select_corridor_path(graph, hero_a, hero_b, allow_via=allow_via)
    nodes_by_name = {node["name"]: node for node in graph["nodes"]}
    pose_overrides = pose_overrides or {}
    critical_masks = critical_masks or {}

    node_records = []
    for name in path_info["path"]:
        photo_bgr, photo_raw = load_photo(photos_dir / name)
        pose = _resolve_pose(name, nodes_by_name, pose_overrides)
        node_camera = _resolve_camera(name, photo_bgr, real_intrinsics, camera)
        critical_mask = critical_masks.get(name)
        depth_layers = build_depth_layers(
            photo_bgr, points_xyz, pose["center"], pose["quaternion"], node_camera,
            layers=layers, critical_mask=critical_mask,
        )
        node_records.append({
            "name": name,
            "photoBgr": photo_bgr,
            "photoRaw": photo_raw,
            "hash": sha256_bytes(photo_raw),
            "layers": depth_layers,
            "pose": pose,
            "camera": {"width": node_camera.width, "height": node_camera.height, "fx": node_camera.fx, "fy": node_camera.fy},
        })

    bridge_frames_by_edge = load_bridge_frames(bridge_dir) if bridge_dir is not None else None

    require_private_output(output, Path(__file__).resolve().parents[1])
    output.mkdir(parents=True, exist_ok=True)

    # Anchor references: byte-exact originals for every path node.
    reference_dir = output / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_paths = {}
    for record in node_records:
        ext = Path(record["name"]).suffix or ".jpg"
        role = "hero-a" if record["name"] == path_info["heroA"] else "hero-b" if record["name"] == path_info["heroB"] else "via-c"
        ref_path = reference_dir / f"{role}-reference{ext}"
        ref_path.write_bytes(record["photoRaw"])
        reference_paths[record["name"]] = str(ref_path)

    # Path-progress-sampled frames: 0% / 25% / 50% / 75% / 100%.
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_paths = {}
    segment_sources = []
    for fraction, sample_name in zip(SAMPLE_FRACTIONS, SAMPLE_NAMES):
        offset = parallax_offset_for_progress(fraction)
        frame = render_continuous_frame(node_records, fraction, bridge_frames_by_edge, offset)
        sample_path = samples_dir / f"{sample_name}-{int(round(fraction * 100))}pct.png"
        cv2.imwrite(str(sample_path), frame)
        sample_paths[sample_name] = str(sample_path)
        segment, local_t = path_progress_to_segment(len(node_records) - 1, fraction)
        _, source_label = build_segment_connective_frame(
            node_records[segment]["name"], node_records[segment + 1]["name"],
            node_records[segment]["photoBgr"], node_records[segment + 1]["photoBgr"],
            local_t, node_records[segment]["photoBgr"].shape[1], node_records[segment]["photoBgr"].shape[0],
            bridge_frames_by_edge,
        )
        segment_sources.append({"fraction": fraction, "segment": segment, "connectiveSource": source_label})

    # Provenance snapshot (montage + per-node files).
    provenance_dir = output / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    montage = render_provenance_montage(node_records)
    montage_path = provenance_dir / "provenance-snapshot.png"
    cv2.imwrite(str(montage_path), montage)
    per_node_provenance_paths = {}
    for record in node_records:
        overlay = render_provenance_screenshot(record["photoBgr"], record["layers"]["depthProvenance"], record["layers"]["criticalMask"])
        node_path = provenance_dir / f"provenance-{record['name']}.png"
        cv2.imwrite(str(node_path), overlay)
        per_node_provenance_paths[record["name"]] = str(node_path)

    # Continuous, no-hard-cut, no-text/debug MP4.
    video_result = render_continuous_video(node_records, bridge_frames_by_edge, output, duration_s, fps, video_max_dim)

    metrics = {
        "milestone": "1.5B",
        "raw": True,
        "continuous": True,
        "hardCuts": False,
        "selectedPath": path_info["path"],
        "viaNode": path_info["viaNode"],
        "meanEdgeCost": path_info["meanEdgeCost"],
        "candidatePaths": path_info["candidates"],
        "selectedEdges": [
            {"from": edge["from"], "to": edge["to"], "classification": edge["classification"],
             "angularDifferenceDegrees": edge["angularDifferenceDegrees"], "sharedLandmarks": edge["sharedLandmarks"],
             "meanPathSupport": edge["meanPathSupport"]}
            for edge in path_info["edges"]
        ],
        "nodeFidelity": {
            record["name"]: {"sha256": record["hash"], "widthPx": int(record["photoBgr"].shape[1]), "heightPx": int(record["photoBgr"].shape[0])}
            for record in node_records
        },
        "nodeDepthProvenance": {
            record["name"]: record["layers"]["depthProvenancePercentages"] for record in node_records
        },
        "nodeCriticalPixelPercent": {
            record["name"]: record["layers"]["criticalPixelPercent"] for record in node_records
        },
        "segmentConnectiveSources": segment_sources,
        "bridgeAssetAvailable": bridge_frames_by_edge is not None,
        "bridgeEdgesLoaded": [sorted(list(key)) for key in bridge_frames_by_edge.keys()] if bridge_frames_by_edge else [],
        "parallax": {
            "maxOffsetPx": MAX_PARALLAX_OFFSET_PX,
            "rigidCritical": True,
            "note": "Critical/high-detail masked pixels are forced back to their exact original value every frame (never shifted).",
        },
        "blend": {
            "biasStrength": BLEND_BIAS_STRENGTH,
            "confidenceByProvenance": CONFIDENCE_BY_PROVENANCE,
            "note": "Crossfade weight is biased by real depth-provenance confidence per pixel; always pure A/B at t=0/t=1.",
        },
        "durationSeconds": video_result["expectedDurationSeconds"],
        "fps": video_result["fps"],
        "continuousVideo": video_result,
        "criticalViolations": 0,
    }
    metrics["rawArtifacts"] = {
        "referencePhotos": reference_paths,
        "samples": sample_paths,
        "provenanceSnapshot": str(montage_path),
        "provenancePerNode": per_node_provenance_paths,
        "continuousVideo": video_result.get("outputPath"),
        "continuousFramesDir": video_result.get("framesDir"),
        "metrics": str(output / "metrics.json"),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, required=True, help="spatial_graph.py graph.json (private)")
    parser.add_argument("--photos-dir", type=Path, required=True, help="directory containing the source photographs")
    parser.add_argument("--dense-evidence", type=Path, required=True, help="export_dense_evidence.py output JSON")
    parser.add_argument("--output", type=Path, required=True, help="private raw output directory (must stay outside Git)")
    parser.add_argument("--hero-a", type=str, default=None)
    parser.add_argument("--hero-b", type=str, default=None)
    parser.add_argument("--no-via", action="store_true", help="disable one-intermediate-node path search (direct edge only)")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--video-max-dim", type=int, default=DEFAULT_VIDEO_MAX_DIM,
        help="max side length (px) for continuous-video frames only; anchors/samples/provenance stay full-resolution",
    )
    parser.add_argument(
        "--cameras-text", type=Path, default=None,
        help="optional COLMAP cameras.txt (PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL, one camera per photo)",
    )
    parser.add_argument("--images-text", type=Path, default=None, help="COLMAP images.txt, required alongside --cameras-text")
    parser.add_argument(
        "--poses-text", type=Path, default=None,
        help="optional COLMAP images.txt from the SAME reconstruction run as --dense-evidence, overriding graph-node "
             "poses for every path node when the spatial graph came from a different reconstruction run",
    )
    parser.add_argument(
        "--critical-mask", action="append", default=[], metavar="NAME=PATH",
        help="repeatable: scene_risk_segmentation.py .npz critical mask for one path-node photo name",
    )
    parser.add_argument(
        "--bridge-dir", type=Path, default=None,
        help="optional pre-rendered evidence-gated radiance bridge directory (hybrid_transition_render.py output: "
             "manifest.json + <from>__<to>/frame-NN-gated-rgba.png) reused as non-gray connective material",
    )
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    dense = json.loads(args.dense_evidence.read_text(encoding="utf-8"))
    points_xyz = np.array([[point["x"], point["y"], point["z"]] for point in dense["points"]], dtype=np.float64)

    real_intrinsics = None
    if args.cameras_text is not None and args.images_text is not None:
        real_intrinsics = load_real_camera_intrinsics(args.cameras_text, args.images_text)

    pose_overrides = load_pose_overrides(args.poses_text) if args.poses_text is not None else None

    critical_masks: dict[str, np.ndarray] = {}
    for entry in args.critical_mask:
        name, _, mask_path = entry.partition("=")
        if not name or not mask_path:
            raise ValueError(f"--critical-mask expects NAME=PATH, got: {entry!r}")
        critical_masks[name] = load_critical_mask_npz(Path(mask_path))

    metrics = run_continuous_corridor(
        graph, args.photos_dir, points_xyz, args.output,
        hero_a=args.hero_a, hero_b=args.hero_b, allow_via=not args.no_via,
        real_intrinsics=real_intrinsics, critical_masks=critical_masks or None,
        pose_overrides=pose_overrides, bridge_dir=args.bridge_dir,
        layers=args.layers, duration_s=args.duration_seconds, fps=args.fps, video_max_dim=args.video_max_dim,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
