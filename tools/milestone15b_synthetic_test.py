#!/usr/bin/env python3
"""Fast, private-data-free regression tests for Milestone 1.5B (continuous
multi-hop photo corridor)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, INFERRED, OBSERVED
from milestone15_photo_corridor import build_depth_layers, default_camera_for_photo, require_private_output
from milestone15b_continuous_corridor import (
    BLEND_BIAS_STRENGTH,
    bridge_frame_for_fraction,
    build_segment_connective_frame,
    composite_parallax_frame_rigid,
    edge_cost,
    interpolate_pose,
    load_bridge_frames,
    parallax_offset_for_progress,
    path_progress_to_segment,
    photo_crossfade_canvas,
    provenance_confidence,
    render_continuous_frame,
    render_continuous_video,
    render_provenance_montage,
    run_continuous_corridor,
    select_corridor_path,
    slerp_quaternion,
    support_aware_blend_weight,
)


def make_synthetic_photo(rng: np.random.Generator, width: int = 200, height: int = 140, seed_shift: int = 0) -> np.ndarray:
    photo = np.clip(rng.normal(130 + seed_shift, 35, (height, width, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(photo, (20, 20), (70, 100), (50 + seed_shift, 120, 190), -1)
    return photo


def make_synthetic_points(rng: np.random.Generator, center: np.ndarray, count: int = 3000) -> np.ndarray:
    forward_depth = rng.uniform(0.6, 6.0, count)
    lateral = rng.uniform(-1.6, 1.6, count)
    vertical = rng.uniform(-1.0, 1.0, count)
    return np.column_stack((center[0] + lateral, center[1] + vertical, center[2] + forward_depth))


# --------------------------------------------------------------------------
# 1. Multi-hop path selection.
# --------------------------------------------------------------------------

def make_via_wins_graph() -> dict:
    """Direct A-B edge has a huge angular change and weak support; A-C and
    C-B are both small angular changes with strong support -- the via path
    through C should clearly win on mean per-edge cost."""
    return {
        "nodes": [
            {"name": "a.jpg", "center": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            {"name": "b.jpg", "center": [2.0, 0.0, 0.0], "quaternion": [0.99, 0.0, 0.1, 0.0]},
            {"name": "c.jpg", "center": [1.0, 0.0, 0.2], "quaternion": [0.98, 0.0, 0.15, 0.0]},
        ],
        "edges": [
            {"from": "a.jpg", "to": "b.jpg", "classification": "strong", "angularDifferenceDegrees": 170.0, "sharedLandmarks": 60, "meanPathSupport": 0.55, "minPathSupport": 0.3},
            {"from": "a.jpg", "to": "c.jpg", "classification": "weak", "angularDifferenceDegrees": 20.0, "sharedLandmarks": 400, "meanPathSupport": 0.8, "minPathSupport": 0.5},
            {"from": "c.jpg", "to": "b.jpg", "classification": "weak", "angularDifferenceDegrees": 18.0, "sharedLandmarks": 380, "meanPathSupport": 0.78, "minPathSupport": 0.5},
        ],
    }


def make_direct_wins_graph() -> dict:
    """Direct A-B edge is small angular change/high support; any via route
    would be strictly worse -- direct should win."""
    return {
        "nodes": [
            {"name": "a.jpg", "center": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            {"name": "b.jpg", "center": [1.0, 0.0, 0.1], "quaternion": [0.99, 0.0, 0.05, 0.0]},
            {"name": "c.jpg", "center": [-3.0, 0.5, 0.5], "quaternion": [0.7, 0.0, 0.7, 0.0]},
        ],
        "edges": [
            {"from": "a.jpg", "to": "b.jpg", "classification": "strong", "angularDifferenceDegrees": 10.0, "sharedLandmarks": 500, "meanPathSupport": 0.85, "minPathSupport": 0.6},
            {"from": "a.jpg", "to": "c.jpg", "classification": "weak", "angularDifferenceDegrees": 160.0, "sharedLandmarks": 5, "meanPathSupport": 0.2, "minPathSupport": 0.1},
            {"from": "c.jpg", "to": "b.jpg", "classification": "weak", "angularDifferenceDegrees": 155.0, "sharedLandmarks": 4, "meanPathSupport": 0.2, "minPathSupport": 0.1},
        ],
    }


def test_edge_cost_monotonicity() -> None:
    base = {"angularDifferenceDegrees": 90.0, "sharedLandmarks": 100, "meanPathSupport": 0.5}
    worse_angle = {**base, "angularDifferenceDegrees": 170.0}
    better_support = {**base, "meanPathSupport": 0.9}
    better_landmarks = {**base, "sharedLandmarks": 900}
    assert edge_cost(worse_angle) > edge_cost(base), "larger angular change must raise cost"
    assert edge_cost(better_support) < edge_cost(base), "higher support must lower cost"
    assert edge_cost(better_landmarks) < edge_cost(base), "more shared landmarks must lower cost"


def test_select_corridor_path_prefers_via_when_genuinely_better() -> None:
    graph = make_via_wins_graph()
    result = select_corridor_path(graph)
    hero_a, hero_b = result["heroA"], result["heroB"]
    assert {hero_a, hero_b} == {"a.jpg", "b.jpg"}
    assert result["path"] == [hero_a, "c.jpg", hero_b], "via-C path should win on mean per-edge cost"
    assert result["viaNode"] == "c.jpg"
    assert len(result["edges"]) == 2
    assert result["candidateCount"] >= 2


def test_select_corridor_path_prefers_direct_when_better() -> None:
    graph = make_direct_wins_graph()
    result = select_corridor_path(graph)
    hero_a, hero_b = result["heroA"], result["heroB"]
    assert result["path"] == [hero_a, hero_b], "direct edge should win when it is clearly better"
    assert result["viaNode"] is None
    assert len(result["edges"]) == 1


def test_select_corridor_path_no_via_flag_forces_direct() -> None:
    graph = make_via_wins_graph()
    result = select_corridor_path(graph, allow_via=False)
    assert result["path"] == [result["heroA"], result["heroB"]], "disabling via search must fall back to the direct edge"


def test_select_corridor_path_rejects_disconnected_pair() -> None:
    graph = {
        "nodes": [
            {"name": "a.jpg", "center": [0, 0, 0], "quaternion": [1, 0, 0, 0]},
            {"name": "b.jpg", "center": [5, 0, 0], "quaternion": [1, 0, 0, 0]},
        ],
        "edges": [{"from": "a.jpg", "to": "b.jpg", "classification": "strong", "angularDifferenceDegrees": 5.0, "sharedLandmarks": 10, "meanPathSupport": 0.5, "minPathSupport": 0.3}],
    }
    try:
        select_corridor_path(graph, hero_a="a.jpg", hero_b="does-not-exist.jpg")
        raise AssertionError("expected ValueError for a nonexistent hero_b")
    except ValueError:
        pass


# --------------------------------------------------------------------------
# Pose interpolation.
# --------------------------------------------------------------------------

def test_slerp_quaternion_endpoints_and_normalization() -> None:
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([0.7071, 0.0, 0.7071, 0.0])
    start = slerp_quaternion(q1, q2, 0.0)
    end = slerp_quaternion(q1, q2, 1.0)
    mid = slerp_quaternion(q1, q2, 0.5)
    assert np.allclose(start, q1, atol=1e-6)
    assert np.allclose(end, q2 / np.linalg.norm(q2), atol=1e-6)
    assert abs(np.linalg.norm(mid) - 1.0) < 1e-9


def test_interpolate_pose_linear_center() -> None:
    pose_a = {"center": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    pose_b = {"center": [4.0, 2.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    mid = interpolate_pose(pose_a, pose_b, 0.5)
    assert np.allclose(mid["center"], [2.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# 2a. Rigid-critical parallax.
# --------------------------------------------------------------------------

def test_rigid_critical_pixels_never_shift() -> None:
    rng = np.random.default_rng(7)
    photo = make_synthetic_photo(rng)
    height, width = photo.shape[:2]
    camera = default_camera_for_photo(width, height)
    points = make_synthetic_points(rng, np.array([0.0, 0.0, 0.0]))
    layers = build_depth_layers(photo, points, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), camera, layers=3)

    critical_mask = np.zeros((height, width), dtype=bool)
    critical_mask[40:80, 60:120] = True

    offset = (10.0, -6.0)
    canvas = composite_parallax_frame_rigid(layers["layers"], critical_mask, width, height, offset, photo)
    assert np.array_equal(canvas[critical_mask][:, :3], photo[critical_mask]), "critical pixels must be rigid (never shifted)"

    # Sanity: without a critical mask, the plain (non-rigid) composite is used and pixels
    # can legitimately differ from the un-shifted photo once a nonzero offset is applied.
    unmasked_canvas = composite_parallax_frame_rigid(layers["layers"], None, width, height, offset, photo)
    assert not np.array_equal(unmasked_canvas[:, :, :3], photo), "a nonzero offset should visibly move at least some non-critical pixels"


# --------------------------------------------------------------------------
# 2b. Support-aware blend weight.
# --------------------------------------------------------------------------

def test_support_aware_blend_weight_pure_at_endpoints() -> None:
    confidence_a = np.full((10, 10), 1.0)
    confidence_b = np.full((10, 10), 0.15)
    weight_start = support_aware_blend_weight(confidence_a, confidence_b, 0.0)
    weight_end = support_aware_blend_weight(confidence_a, confidence_b, 1.0)
    assert np.allclose(weight_start, 0.0), "t=0 must always be pure A regardless of confidence bias"
    assert np.allclose(weight_end, 1.0), "t=1 must always be pure B regardless of confidence bias"


def test_support_aware_blend_weight_biases_towards_confident_side() -> None:
    depth_provenance_a = np.full((10, 10), OBSERVED, dtype=np.uint8)
    depth_provenance_b = np.full((10, 10), ABSENT, dtype=np.uint8)
    confidence_a = provenance_confidence(depth_provenance_a)
    confidence_b = provenance_confidence(depth_provenance_b)
    biased = support_aware_blend_weight(confidence_a, confidence_b, 0.5)
    plain = 0.5
    assert (biased < plain).all(), "mid-corridor weight should be pulled towards the more confident (A) side"

    biased_reverse = support_aware_blend_weight(confidence_b, confidence_a, 0.5)
    assert (biased_reverse > plain).all(), "mid-corridor weight should be pulled towards the more confident (B) side"


# --------------------------------------------------------------------------
# 3. Non-gray connective space.
# --------------------------------------------------------------------------

def test_photo_crossfade_canvas_is_non_gray_and_derived_from_real_photos() -> None:
    rng = np.random.default_rng(3)
    photo_u = make_synthetic_photo(rng, seed_shift=0)
    photo_v = make_synthetic_photo(rng, seed_shift=60)
    canvas = photo_crossfade_canvas(photo_u, photo_v, 0.5, 160, 120)
    assert canvas.shape == (120, 160, 3)
    # Not a flat single-color canvas: real photo texture must still show through.
    assert canvas.std() > 3.0, "connective canvas must retain real photo texture variation, not be flat"


def test_load_bridge_frames_and_lookup() -> None:
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    try:
        _load_bridge_frames_and_lookup_body(tmp_path)
    finally:
        tmp_dir.cleanup()


def _load_bridge_frames_and_lookup_body(tmp_path: Path) -> None:
    bridge_dir = tmp_path / "bridge"
    edge_dir = bridge_dir / "u.jpg__v.jpg"
    edge_dir.mkdir(parents=True)
    manifest = {"edges": [{"from": "u.jpg", "to": "v.jpg", "frames": [
        {"frame": 0, "fraction": 0.0, "meanSupportScore": 0.4},
        {"frame": 1, "fraction": 0.5, "meanSupportScore": 0.6},
        {"frame": 2, "fraction": 1.0, "meanSupportScore": 0.5},
    ]}]}
    (bridge_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for index, fraction in ((0, 0.0), (1, 0.5), (2, 1.0)):
        rgba = np.full((20, 20, 4), int(fraction * 200) + 10, dtype=np.uint8)
        rgba[:, :, 3] = 255
        cv2.imwrite(str(edge_dir / f"frame-{index:02d}-gated-rgba.png"), rgba)

    loaded = load_bridge_frames(bridge_dir)
    assert loaded is not None
    key = frozenset({"u.jpg", "v.jpg"})
    assert key in loaded
    assert len(loaded[key]) == 3

    nearest = bridge_frame_for_fraction(loaded[key], 0.4)
    assert nearest.shape == (20, 20, 4)

    assert load_bridge_frames(tmp_path / "does-not-exist") is None


def test_build_segment_connective_frame_prefers_bridge_when_available() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bridge_dir = tmp_path / "bridge"
        edge_dir = bridge_dir / "u.jpg__v.jpg"
        edge_dir.mkdir(parents=True)
        manifest = {"edges": [{"from": "u.jpg", "to": "v.jpg", "frames": [{"frame": 0, "fraction": 0.5, "meanSupportScore": 0.7}]}]}
        (bridge_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rgba = np.zeros((20, 20, 4), dtype=np.uint8)
        rgba[:, :, 0] = 10   # blue channel marker
        rgba[:, :, 3] = 255  # fully opaque -> should dominate the composite
        cv2.imwrite(str(edge_dir / "frame-00-gated-rgba.png"), rgba)
        bridge_frames = load_bridge_frames(bridge_dir)

        rng = np.random.default_rng(9)
        photo_u = make_synthetic_photo(rng)
        photo_v = make_synthetic_photo(rng, seed_shift=80)

        canvas_with_bridge, source_with = build_segment_connective_frame("u.jpg", "v.jpg", photo_u, photo_v, 0.5, 20, 20, bridge_frames)
        canvas_without, source_without = build_segment_connective_frame("u.jpg", "v.jpg", photo_u, photo_v, 0.5, 20, 20, None)
        assert source_with == "hybrid-bridge-real-render"
        assert source_without == "photo-crossfade-shell"
        assert not np.array_equal(canvas_with_bridge, canvas_without)


# --------------------------------------------------------------------------
# Continuous frame/video rendering.
# --------------------------------------------------------------------------

def test_path_progress_to_segment_mapping() -> None:
    assert path_progress_to_segment(2, 0.0) == (0, 0.0)
    assert path_progress_to_segment(2, 1.0) == (1, 1.0)
    segment, local_t = path_progress_to_segment(2, 0.25)
    assert segment == 0 and abs(local_t - 0.5) < 1e-9
    segment, local_t = path_progress_to_segment(2, 0.75)
    assert segment == 1 and abs(local_t - 0.5) < 1e-9


def test_parallax_offset_bounded_and_smooth() -> None:
    from milestone15b_continuous_corridor import MAX_PARALLAX_OFFSET_PX
    offsets = [parallax_offset_for_progress(u) for u in np.linspace(0.0, 1.0, 50)]
    for dx, dy in offsets:
        assert abs(dx) <= MAX_PARALLAX_OFFSET_PX + 1e-6
        assert abs(dy) <= MAX_PARALLAX_OFFSET_PX * 0.4 + 1e-6
    # No hard jumps: consecutive frames must differ only a little.
    for (dx1, dy1), (dx2, dy2) in zip(offsets, offsets[1:]):
        assert abs(dx1 - dx2) < 3.0 and abs(dy1 - dy2) < 3.0


def _make_two_node_records(rng: np.random.Generator) -> list[dict]:
    photo_u = make_synthetic_photo(rng, seed_shift=0)
    photo_v = make_synthetic_photo(rng, seed_shift=50)
    height, width = photo_u.shape[:2]
    camera = default_camera_for_photo(width, height)
    points = make_synthetic_points(rng, np.array([0.0, 0.0, 0.0]))
    layers_u = build_depth_layers(photo_u, points, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]), camera, layers=3)
    layers_v = build_depth_layers(photo_v, points, np.array([0.3, 0.0, 0.1]), np.array([0.99, 0.0, 0.05, 0.0]), camera, layers=3)
    return [
        {"name": "u.jpg", "photoBgr": photo_u, "layers": layers_u},
        {"name": "v.jpg", "photoBgr": photo_v, "layers": layers_v},
    ]


def test_render_continuous_frame_is_pure_anchor_at_endpoints() -> None:
    rng = np.random.default_rng(21)
    node_records = _make_two_node_records(rng)
    frame_start = render_continuous_frame(node_records, 0.0, None, parallax_offset_for_progress(0.0))
    frame_end = render_continuous_frame(node_records, 1.0, None, parallax_offset_for_progress(1.0))
    assert frame_start.shape[:2] == node_records[0]["photoBgr"].shape[:2]
    # At t=0 with zero parallax offset the render must match the real photo exactly (no blend, no shift).
    assert parallax_offset_for_progress(0.0) == (0.0, 0.0)
    assert np.array_equal(frame_start, node_records[0]["photoBgr"]), "t=0 at zero offset must equal the untouched real anchor photo"
    assert frame_end.shape[:2] == node_records[1]["photoBgr"].shape[:2]


def test_render_continuous_video_produces_frame_sequence() -> None:
    rng = np.random.default_rng(31)
    node_records = _make_two_node_records(rng)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result = render_continuous_video(node_records, None, tmp_path, duration_s=10.0, fps=24)
        assert result["frameCount"] == 240
        assert result["fps"] == 24
        assert 9.9 <= result["expectedDurationSeconds"] <= 20.1
        frames_dir = Path(result["framesDir"])
        written = sorted(frames_dir.glob("frame-*.png"))
        assert len(written) == 240
        assert result["style"] == "continuous, no hard cuts, no text/debug overlay"
        # Either ffmpeg produced a real mp4, or it failed closed with an explicit blocker -- never silently nothing.
        assert result["produced"] or result["blocker"]


def test_render_provenance_montage_shape() -> None:
    rng = np.random.default_rng(41)
    node_records = _make_two_node_records(rng)
    montage = render_provenance_montage(node_records)
    assert montage.shape[0] == min(r["photoBgr"].shape[0] for r in node_records)
    assert montage.shape[1] >= node_records[0]["photoBgr"].shape[1]


# --------------------------------------------------------------------------
# End-to-end orchestration.
# --------------------------------------------------------------------------

def test_run_continuous_corridor_end_to_end() -> None:
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp_dir.name)
    try:
        _run_continuous_corridor_end_to_end_body(tmp_path)
    finally:
        tmp_dir.cleanup()


def _run_continuous_corridor_end_to_end_body(tmp_path: Path) -> None:
    rng = np.random.default_rng(55)
    graph = make_via_wins_graph()
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    photo_bgr_by_name = {}
    for name, shift in (("a.jpg", 0), ("b.jpg", 90), ("c.jpg", 45)):
        photo = make_synthetic_photo(rng, seed_shift=shift)
        cv2.imwrite(str(photos_dir / name), photo)
        photo_bgr_by_name[name] = photo
    points = make_synthetic_points(rng, np.array([0.0, 0.0, 0.0]), count=6000)

    output = tmp_path / "output" / "corridor"
    metrics = run_continuous_corridor(
        graph, photos_dir, points, output,
        duration_s=10.0, fps=24,
    )

    assert metrics["milestone"] == "1.5B"
    assert metrics["viaNode"] == "c.jpg"
    assert set(metrics["selectedPath"]) == {"a.jpg", "b.jpg", "c.jpg"}
    assert metrics["selectedPath"][1] == "c.jpg"
    hero_a_name = metrics["selectedPath"][0]
    assert metrics["hardCuts"] is False
    assert metrics["continuous"] is True
    assert set(metrics["nodeFidelity"].keys()) == {"a.jpg", "b.jpg", "c.jpg"}
    assert (output / "reference" / "hero-a-reference.jpg").exists()
    assert (output / "reference" / "hero-b-reference.jpg").exists()
    assert (output / "reference" / "via-c-reference.jpg").exists()
    for sample_path in metrics["rawArtifacts"]["samples"].values():
        assert Path(sample_path).exists()
    assert Path(metrics["rawArtifacts"]["provenanceSnapshot"]).exists()
    assert (output / "metrics.json").exists()

    # Fidelity check: whichever real photo ended up as hero A, its reference bytes
    # must be byte-identical to that source file on disk.
    original_bytes = (photos_dir / hero_a_name).read_bytes()
    written_bytes = (output / "reference" / "hero-a-reference.jpg").read_bytes()
    assert original_bytes == written_bytes


def test_run_continuous_corridor_rejects_in_repo_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rng = np.random.default_rng(61)
        graph = make_direct_wins_graph()
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()
        for name, shift in (("a.jpg", 0), ("b.jpg", 60)):
            cv2.imwrite(str(photos_dir / name), make_synthetic_photo(rng, seed_shift=shift))
        points = make_synthetic_points(rng, np.array([0.0, 0.0, 0.0]))
        repo_root = Path(__file__).resolve().parents[1]
        try:
            run_continuous_corridor(graph, photos_dir, points, repo_root / "tools", duration_s=10.0, fps=24)
            raise AssertionError("expected ValueError for an in-repo output directory")
        except ValueError:
            pass


TEST_FUNCTIONS = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]


def main() -> None:
    passed = 0
    for test_fn in TEST_FUNCTIONS:
        argcount = test_fn.__code__.co_argcount
        if argcount == 0:
            test_fn()
        else:
            with tempfile.TemporaryDirectory() as tmp:
                test_fn(Path(tmp))
        passed += 1
        print(f"PASS {test_fn.__name__}")
    print(f"\n{passed}/{len(TEST_FUNCTIONS)} tests passed")


if __name__ == "__main__":
    main()
