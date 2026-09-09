"""Milestone 0.7: spatial-continuity transitions between real captured photos.

This tool does NOT synthesize any new scene content. Every transition method
implemented here starts and ends on an untouched source photograph and only
ever displays pixels that came from one of those two photographs. "Movement"
is communicated by (a) 2D perspective warps of the *existing* source pixels
driven by feature correspondences between the pair, and/or (b) opacity
blending between the two photographs. Regions with no supporting evidence
(no nearby matched feature) fade toward transparency rather than being
filled in.

Four methods are produced per selected photo pair, matching the Milestone 0.7
brief's baseline/candidate structure:

  baseline_hard_cut   - Photo A, then instantly Photo B. No interpolation.
  baseline_crossfade  - Plain opacity blend from Photo A to Photo B.
  candidate_geometry  - Geometry-aligned camera move: Photo A is progressively
                        perspective-warped (using the recovered A->B
                        homography from real matched features) to visually
                        suggest camera translation/rotation toward B, then
                        hands off to Photo B. Areas that leave the frame
                        during the warp fade to transparent; no new pixels
                        are invented.
  candidate_warp      - Correspondence warp: Photo A is warped toward B using
                        the same homography, but every pixel's opacity is
                        additionally modulated by a support field built from
                        the *density of inlier feature matches* near that
                        pixel. Pixels far from any matched landmark fade out
                        rather than being stretched/hallucinated. If the
                        match is not credible (too few/weak inliers), this
                        method is marked not-credible for that pair and a
                        crossfade fallback is produced instead.

The recovered 3D camera-pose delta (world-space translation + forward-vector
rotation, already present in the Milestone 0.6 spatial graph) is used only as
a *diagnostic*: it is compared against the direction implied by the 2D
homography to sanity-check that the visual motion direction agrees with the
recovered spatial relationship. It is not used to reproject/hallucinate
pixels.

All image I/O, feature matching, and frame rendering happens on whatever
private paths are passed on the CLI. This script itself contains no photos,
poses, or other identifying data and is safe to commit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Tunable thresholds -----------------------------------------------------

MIN_SHARED_LANDMARKS_FOR_CANDIDATE = 150
MIN_CAMERA_DISPLACEMENT_FOR_CANDIDATE = 1.0  # SfM units (unscaled), not meters
MAX_ANGLE_FOR_CANDIDATE_DEGREES = 100.0  # beyond this, correspondence gets unreliable

# Correspondence-warp credibility gate (Candidate D)
MIN_INLIERS_FOR_CREDIBLE_WARP = 25
MIN_INLIER_RATIO_FOR_CREDIBLE_WARP = 0.35

N_FRAMES = 18


# --------------------------------------------------------------------------
# Pair selection
# --------------------------------------------------------------------------

def score_edge(edge: dict) -> float:
    """Score a graph edge for suitability as a Milestone 0.7 transition pair.

    Favors: many shared landmarks (recognizable shared objects/overlap),
    meaningful camera displacement (not a near-duplicate pose), and an
    angular change that is large enough to be a "real" viewpoint change but
    not so large that correspondence becomes unreliable.
    """
    landmarks = edge["sharedLandmarks"]
    displacement = edge["cameraDistance"]
    angle = edge["angularDifferenceDegrees"]

    if landmarks < MIN_SHARED_LANDMARKS_FOR_CANDIDATE:
        return -1.0
    if displacement < MIN_CAMERA_DISPLACEMENT_FOR_CANDIDATE:
        return -1.0
    if angle > MAX_ANGLE_FOR_CANDIDATE_DEGREES:
        return -1.0

    # Reward landmark count (log-scaled so a handful of very-high-landmark
    # edges don't totally dominate), reward moderate angular change
    # (peaked around 45 degrees -- a real but not extreme viewpoint change),
    # and mildly reward larger displacement (a more meaningful move through
    # space) with diminishing returns.
    landmark_term = math.log1p(landmarks)
    angle_term = 1.0 - abs(angle - 45.0) / 90.0
    displacement_term = math.log1p(displacement)
    return 2.0 * landmark_term + 1.5 * angle_term + 0.5 * displacement_term


def select_pairs(graph: dict, count: int) -> list[dict]:
    scored = []
    for edge in graph["edges"]:
        s = score_edge(edge)
        if s > 0:
            scored.append((s, edge))
    scored.sort(key=lambda item: -item[0])

    selected = []
    used_nodes: set[str] = set()
    for s, edge in scored:
        # Prefer diversity: don't pick two pairs that share both endpoints
        # with an already-selected pair, so the 2-3 chosen pairs sample
        # different parts of the room rather than one cluster.
        pair_nodes = {edge["from"], edge["to"]}
        if selected and pair_nodes.issubset(used_nodes) and len(pair_nodes & used_nodes) == 2:
            overlap_already_covered = any(
                pair_nodes == {e["from"], e["to"]} for _, e in [(0, sel) for sel in selected]
            )
            if overlap_already_covered:
                continue
        selected.append(edge)
        used_nodes |= pair_nodes
        if len(selected) >= count:
            break
    return selected


# --------------------------------------------------------------------------
# Feature matching / homography
# --------------------------------------------------------------------------

def match_features(img_a: np.ndarray, img_b: np.ndarray):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    detector = cv2.SIFT_create(nfeatures=4000)
    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    homography, inlier_mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 4.0)
    if homography is None:
        return None
    inlier_mask = inlier_mask.ravel().astype(bool)

    return {
        "homography": homography,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "inlier_mask": inlier_mask,
        "n_matches": len(good),
        "n_inliers": int(inlier_mask.sum()),
        "inlier_ratio": float(inlier_mask.sum()) / len(good),
    }


def interpolate_homography(homography: np.ndarray, t: float) -> np.ndarray:
    """Linearly blend a homography with identity and renormalize.

    This is an approximate ("Ken Burns"-style) interpolation, not a true
    projective-motion interpolation, but it is visually reasonable for the
    moderate camera displacements selected here and never introduces new
    scene content -- it only ever repositions existing source pixels.
    """
    identity = np.eye(3, dtype=np.float64)
    blended = (1.0 - t) * identity + t * homography
    if abs(blended[2, 2]) > 1e-8:
        blended = blended / blended[2, 2]
    return blended


def support_field(shape: tuple[int, int], inlier_points: np.ndarray) -> np.ndarray:
    """Build a smooth 0..1 support map from inlier keypoint density.

    Pixels far from any matched/inlier landmark get low support and will be
    faded out rather than stretched or hallucinated.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y in inlier_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            mask[yi, xi] = 255
    if not mask.any():
        return np.zeros((h, w), dtype=np.float32)

    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    # Support decays smoothly; points within ~120px of an inlier landmark
    # keep strong support, decaying to ~0 by ~400px.
    support = np.exp(-(dist / 180.0) ** 2)
    return support.astype(np.float32)


# --------------------------------------------------------------------------
# Geometry-delta diagnostic (recovered 3D pose vs. 2D homography agreement)
# --------------------------------------------------------------------------

def geometry_consistency_check(node_a: dict, node_b: dict) -> dict:
    center_a = np.array(node_a["center"])
    center_b = np.array(node_b["center"])
    forward_a = np.array(node_a["forward"])
    forward_a = forward_a / np.linalg.norm(forward_a)

    translation = center_b - center_a
    translation_norm = np.linalg.norm(translation)
    if translation_norm < 1e-9:
        forward_component = 0.0
    else:
        forward_component = float(np.dot(translation / translation_norm, forward_a))

    # forward_component > 0 roughly means "B is further along where A was
    # looking" (a forward move); < 0 means B is behind A's view direction.
    return {
        "worldTranslationNorm": float(translation_norm),
        "forwardAlignment": forward_component,
        "interpretation": (
            "B is ahead of A's view direction (forward move)"
            if forward_component > 0.25
            else "B is behind A's view direction (backward move)"
            if forward_component < -0.25
            else "B is mostly lateral/orbital relative to A's view direction"
        ),
    }


# --------------------------------------------------------------------------
# Frame rendering for each method
# --------------------------------------------------------------------------

def render_hard_cut(img_a, img_b, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "frame_00.png"), img_a)
    cv2.imwrite(str(out_dir / "frame_01.png"), img_b)
    return {"frameCount": 2}


def render_crossfade(img_a, img_b, out_dir: Path, n_frames: int = N_FRAMES):
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h))
    for i in range(n_frames):
        t = i / (n_frames - 1)
        frame = cv2.addWeighted(img_a, 1.0 - t, img_b_resized, t, 0)
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.png"), frame)
    return {"frameCount": n_frames}


def render_geometry_move(img_a, img_b, homography, out_dir: Path, n_frames: int = N_FRAMES):
    """Progressively warp Photo A toward B's framing, then hand off to B.

    The first ~65% of frames only warp Photo A's own pixels (a "camera
    move" over the still-mostly-intact outgoing photo); the remaining
    frames crossfade quickly into the untouched Photo B. Regions that warp
    outside the original frame become transparent (alpha channel), never
    invented content.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h))

    handoff_at = 0.65
    valid_mask_a = np.ones((h, w), dtype=np.uint8) * 255

    stats = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        warp_t = min(t, handoff_at) / handoff_at
        h_t = interpolate_homography(homography, warp_t * 0.8)  # cap warp strength; A stays mostly intact

        warped = cv2.warpPerspective(img_a, h_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_mask = cv2.warpPerspective(valid_mask_a, h_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if t <= handoff_at:
            frame_bgr = warped
            alpha = warped_mask
        else:
            fade_t = (t - handoff_at) / (1.0 - handoff_at)
            frame_bgr = cv2.addWeighted(warped, 1.0 - fade_t, img_b_resized, fade_t, 0)
            alpha = np.maximum(warped_mask, np.uint8(fade_t * 255))

        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.png"), bgra)
        stats.append({
            "t": t,
            "unsupportedFraction": float(1.0 - (warped_mask > 0).mean()),
        })
    return {"frameCount": n_frames, "handoffAt": handoff_at, "perFrame": stats}


def render_correspondence_warp(img_a, img_b, match, out_dir: Path, n_frames: int = N_FRAMES):
    """Warp Photo A toward B using matched inlier landmarks, fading any
    pixel that is not near a supporting match. Falls back to reporting
    not-credible if the match quality is too low, per the milestone's
    failure conditions (do not force a rubbery warp).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h))

    credible = (
        match["n_inliers"] >= MIN_INLIERS_FOR_CREDIBLE_WARP
        and match["inlier_ratio"] >= MIN_INLIER_RATIO_FOR_CREDIBLE_WARP
    )

    if not credible:
        # Fallback: honest crossfade, clearly labeled not-credible.
        result = render_crossfade(img_a, img_b, out_dir, n_frames)
        result["credible"] = False
        result["reason"] = (
            f"inliers={match['n_inliers']} (min {MIN_INLIERS_FOR_CREDIBLE_WARP}), "
            f"inlier_ratio={match['inlier_ratio']:.2f} (min {MIN_INLIER_RATIO_FOR_CREDIBLE_WARP})"
        )
        return result

    homography = match["homography"]
    inlier_pts_a = match["pts_a"][match["inlier_mask"]]
    # Support is built once in A's original pixel space, then re-warped by
    # the *same* per-frame homography as the image itself each frame, so the
    # support field tracks the moving landmarks instead of staying anchored
    # to A's original (now-stale) pixel coordinates.
    support_in_a = support_field((h, w), inlier_pts_a)

    stats = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        h_t = interpolate_homography(homography, t)
        warped = cv2.warpPerspective(img_a, h_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        warped_valid = cv2.warpPerspective(
            np.ones((h, w), dtype=np.uint8) * 255, h_t, (w, h),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.float32) / 255.0
        warped_support = cv2.warpPerspective(
            support_in_a, h_t, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        )

        # Support fades in favor of destination photo as t increases and as
        # local match density drops -- never invents unsupported pixels.
        local_alpha = warped_valid * warped_support * (1.0 - 0.6 * t)
        frame_bgr = (warped.astype(np.float32) * local_alpha[..., None]
                     + img_b_resized.astype(np.float32) * (1.0 - local_alpha[..., None]))
        frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)

        bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255  # final composited frame always fully opaque (B backs unsupported regions)
        cv2.imwrite(str(out_dir / f"frame_{i:02d}.png"), bgra)
        stats.append({"t": t, "meanSupport": float(local_alpha.mean())})

    return {"frameCount": n_frames, "credible": True, "perFrame": stats,
            "nInliers": match["n_inliers"], "inlierRatio": match["inlier_ratio"]}


def render_landmark_overlay(img_a, img_b, match, out_path: Path):
    """Diagnostic-only visualization of matched inlier landmarks (Phase 3)."""
    kp_a = [cv2.KeyPoint(x, y, 5) for x, y in match["pts_a"][match["inlier_mask"]]]
    kp_b = [cv2.KeyPoint(x, y, 5) for x, y in match["pts_b"][match["inlier_mask"]]]
    dmatches = [cv2.DMatch(i, i, 0) for i in range(len(kp_a))]
    vis = cv2.drawMatches(img_a, kp_a, img_b, kp_b, dmatches, None,
                           matchColor=(0, 220, 0), singlePointColor=(0, 0, 255),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Path to private graph.json (with poses)")
    parser.add_argument("--images", required=True, help="Path to private source-photo directory")
    parser.add_argument("--output", required=True, help="Private output directory for frames/diagnostics")
    parser.add_argument("--pairs", type=int, default=3, help="Number of pairs to select (2-3)")
    args = parser.parse_args()

    graph = json.loads(Path(args.graph).read_text())
    nodes_by_name = {n["name"]: n for n in graph["nodes"]}
    images_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_pairs(graph, args.pairs)
    manifest = {"pairs": []}

    for idx, edge in enumerate(selected):
        name_a, name_b = edge["from"], edge["to"]
        pair_dir = out_dir / f"pair_{idx:02d}"
        img_a = cv2.imread(str(images_dir / name_a))
        img_b = cv2.imread(str(images_dir / name_b))
        if img_a is None or img_b is None:
            print(f"Skipping pair {name_a} <-> {name_b}: could not read source images")
            continue

        match = match_features(img_a, img_b)
        pair_entry = {
            "pairIndex": idx,
            "nameA": name_a,
            "nameB": name_b,
            "sharedLandmarksInGraph": edge["sharedLandmarks"],
            "cameraDistance": edge["cameraDistance"],
            "angularDifferenceDegrees": edge["angularDifferenceDegrees"],
            "selectionScore": score_edge(edge),
            "geometryConsistency": geometry_consistency_check(nodes_by_name[name_a], nodes_by_name[name_b]),
        }

        if match is None:
            pair_entry["error"] = "Feature matching failed (too few matches/keypoints)"
            manifest["pairs"].append(pair_entry)
            continue

        pair_entry["featureMatch"] = {
            "nMatches": match["n_matches"],
            "nInliers": match["n_inliers"],
            "inlierRatio": match["inlier_ratio"],
        }

        pair_entry["methods"] = {
            "baseline_hard_cut": render_hard_cut(img_a, img_b, pair_dir / "baseline_hard_cut"),
            "baseline_crossfade": render_crossfade(img_a, img_b, pair_dir / "baseline_crossfade"),
            "candidate_geometry": render_geometry_move(img_a, img_b, match["homography"], pair_dir / "candidate_geometry"),
            "candidate_warp": render_correspondence_warp(img_a, img_b, match, pair_dir / "candidate_warp"),
        }

        render_landmark_overlay(img_a, img_b, match, pair_dir / "landmark_overlay.png")

        manifest["pairs"].append(pair_entry)
        print(f"Pair {idx}: {name_a} <-> {name_b} "
              f"(landmarks={edge['sharedLandmarks']}, inliers={match['n_inliers']}, "
              f"warp_credible={pair_entry['methods']['candidate_warp']['credible']})")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest and frames to {out_dir}")


if __name__ == "__main__":
    main()
