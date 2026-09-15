#!/usr/bin/env python3
"""Fast, private-data-free regression tests for Milestone 1.5 (photo corridor)."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, IMAGINED, INFERRED, OBSERVED
from milestone15_photo_corridor import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    build_connective_shell,
    build_depth_layers,
    build_walkthrough,
    composite_offset_frame,
    default_camera_for_photo,
    require_private_output,
    select_hero_pair,
    sha256_bytes,
)


def make_synthetic_graph() -> dict:
    """Three-node spatial graph: 'a' has the strongest evidence, one strong
    edge to 'b', one weak edge to 'c'. Procedural only, no real capture."""
    return {
        "nodes": [
            {"name": "a.jpg", "center": [0.0, 0.0, 0.0], "quaternion": [1.0, 0.0, 0.0, 0.0], "forward": [0, 0, 1]},
            {"name": "b.jpg", "center": [1.2, 0.0, 0.2], "quaternion": [0.99, 0.0, 0.05, 0.0], "forward": [0, 0, 1]},
            {"name": "c.jpg", "center": [-3.0, 0.1, 0.5], "quaternion": [1.0, 0.0, 0.0, 0.0], "forward": [0, 0, 1]},
        ],
        "edges": [
            {"from": "a.jpg", "to": "b.jpg", "classification": "strong", "meanPathSupport": 0.91, "minPathSupport": 0.6, "sharedLandmarks": 40},
            {"from": "a.jpg", "to": "c.jpg", "classification": "weak", "meanPathSupport": 0.30, "minPathSupport": 0.1, "sharedLandmarks": 10},
        ],
    }


def make_synthetic_photo(rng: np.random.Generator, width: int = 240, height: int = 160) -> np.ndarray:
    photo = np.clip(rng.normal(140, 40, (height, width, 3)), 0, 255).astype(np.uint8)
    cv2.rectangle(photo, (30, 30), (90, 120), (60, 140, 200), -1)  # a synthetic "near" object
    return photo


def make_synthetic_points(rng: np.random.Generator, center: np.ndarray, count: int = 4000) -> np.ndarray:
    """Points spread across a range of depths in front of the given camera center."""
    forward_depth = rng.uniform(0.6, 6.0, count)
    lateral = rng.uniform(-1.6, 1.6, count)
    vertical = rng.uniform(-1.0, 1.0, count)
    points = np.column_stack((center[0] + lateral, center[1] + vertical, center[2] + forward_depth))
    return points


def test_hero_pair_selection() -> None:
    graph = make_synthetic_graph()
    hero_a, hero_b, edge = select_hero_pair(graph)
    assert hero_a == "a.jpg", "node with the strongest evidence should be selected as hero A"
    assert hero_b == "b.jpg", "hero A's strong edge partner should be selected over its weak edge"
    assert edge["classification"] == "strong"

    # Explicit override still resolves the matching edge.
    hero_a, hero_b, edge = select_hero_pair(graph, hero_a="a.jpg", hero_b="c.jpg")
    assert (hero_a, hero_b) == ("a.jpg", "c.jpg")
    assert edge["classification"] == "weak"

    try:
        select_hero_pair({"nodes": [], "edges": []})
        raise AssertionError("expected ValueError for an edgeless graph")
    except ValueError:
        pass


def test_output_safety() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        require_private_output(repo_root / "tools", repo_root)
        raise AssertionError("expected rejection of an in-repo output directory")
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        require_private_output(Path(tmp), repo_root)  # must not raise


def test_depth_layers_preserve_photo_fidelity() -> None:
    rng = np.random.default_rng(15)
    photo = make_synthetic_photo(rng)
    height, width = photo.shape[:2]
    camera = default_camera_for_photo(width, height)
    points = make_synthetic_points(rng, np.array([0.0, 0.0, 0.0]))
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])

    result = build_depth_layers(photo, points, center, quaternion, camera, layers=3)
    layers = result["layers"]
    assert len(layers) == 3
    assert sum(record["pixelCount"] for record in layers) <= height * width

    covered = np.zeros((height, width), dtype=bool)
    for record in layers:
        rgba = record["rgba"]
        opaque = rgba[:, :, 3] > 0
        # Doctrine v2 fidelity guarantee: every opaque hero-layer pixel is
        # byte-identical to the original captured photograph.
        assert np.array_equal(rgba[opaque][:, :3], photo[opaque]), "hero layer pixels must be byte-exact to the source photo"
        covered |= opaque
    # No holes: every pixel of the real photo remains visible in some layer.
    assert covered.all(), "every hero photo pixel must remain visible across the depth layers"

    percentages = result["depthProvenancePercentages"]
    total = sum(percentages.values())
    assert abs(total - 100.0) < 1e-6
    assert percentages["depthObservedPercent"] > 0.0

    try:
        build_depth_layers(photo, points, center, quaternion, camera, layers=1)
        raise AssertionError("expected ValueError for fewer than 2 layers")
    except ValueError:
        pass


def test_depth_layers_reject_no_projected_evidence() -> None:
    rng = np.random.default_rng(3)
    photo = make_synthetic_photo(rng)
    camera = default_camera_for_photo(photo.shape[1], photo.shape[0])
    behind_camera_points = np.array([[0.0, 0.0, -5.0]] * 10)  # entirely behind the camera.
    try:
        build_depth_layers(photo, behind_camera_points, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), camera)
        raise AssertionError("expected ValueError when no evidence projects into the photo")
    except ValueError:
        pass


def test_connective_shell_is_subordinate_and_imagined() -> None:
    rng = np.random.default_rng(21)
    hero_a = make_synthetic_photo(rng)
    hero_b = make_synthetic_photo(rng)
    shell = build_connective_shell(hero_a, hero_b, hero_a.shape[1], hero_a.shape[0], engine="deterministic")

    assert shell["imaginedPercent"] == 100.0, "connective shell must be entirely IMAGINED, never claimed as evidence"
    assert shell["observedPercent"] == 0.0 and shell["reconstructedPercent"] == 0.0
    assert shell["subordinate"]["opacity"] < 1.0, "shell opacity must stay below hero (full) opacity"
    assert shell["subordinate"]["renderScale"] < 1.0, "shell must render at lower resolution than hero photos"
    assert shell["engineStatus"]["selectedEngine"] == "deterministic"

    # Requesting the learned engine without a runnable model must fail closed,
    # not silently fabricate content over the shell.
    shell_learned = build_connective_shell(hero_a, hero_b, hero_a.shape[1], hero_a.shape[0], engine="learned")
    assert shell_learned["engineStatus"]["selectedEngine"] == "fail-closed-learned"
    assert shell_learned["engineStatus"]["blocker"] is not None


def test_connective_shell_learned_runner_seam() -> None:
    """A stub runner (no real model download) proves the constrained-refinement
    path is reused correctly and stays strictly IMAGINED provenance."""
    rng = np.random.default_rng(22)
    hero_a = make_synthetic_photo(rng)
    hero_b = make_synthetic_photo(rng)

    def stub_runner(resized_rgb, resized_mask, prompt, context_rgb=None, strength=1.0):
        return resized_rgb  # deterministic passthrough stub; no network/model.

    shell = build_connective_shell(
        hero_a, hero_b, hero_a.shape[1], hero_a.shape[0], engine="learned", runner=stub_runner,
    )
    assert shell["engineStatus"]["selectedEngine"] == "learned-diffusion"
    assert shell["imaginedPercent"] == 100.0


def test_composite_offset_frame_shifts_layers() -> None:
    rng = np.random.default_rng(15)
    photo = make_synthetic_photo(rng)
    height, width = photo.shape[:2]
    camera = default_camera_for_photo(width, height)
    points = make_synthetic_points(rng, np.zeros(3))
    result = build_depth_layers(photo, points, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), camera, layers=3)
    centered = composite_offset_frame(result["layers"], width, height, (0, 0))
    shifted = composite_offset_frame(result["layers"], width, height, (24, 0))
    assert not np.array_equal(centered, shifted), "a nonzero screen offset must visibly move parallax layers"


def test_walkthrough_duration_clamped_and_raw() -> None:
    rng = np.random.default_rng(15)
    hero_a_photo = make_synthetic_photo(rng)
    hero_b_photo = make_synthetic_photo(rng)
    camera = default_camera_for_photo(hero_a_photo.shape[1], hero_a_photo.shape[0])
    points = make_synthetic_points(rng, np.zeros(3))
    hero_a_layers = build_depth_layers(hero_a_photo, points, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), camera)
    hero_b_layers = build_depth_layers(hero_b_photo, points, np.array([1.2, 0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]), camera)
    shell = build_connective_shell(hero_a_photo, hero_b_photo, hero_a_photo.shape[1], hero_a_photo.shape[0])

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        metrics = build_walkthrough(output, "a.jpg", "b.jpg", hero_a_layers, hero_b_layers, shell, duration_s=999.0)
        assert MIN_DURATION_S <= metrics["durationSeconds"] <= MAX_DURATION_S, "raw walkthrough duration must clamp to 10-20s"
        assert metrics["raw"] is True and metrics["frozen"] is False
        assert metrics["screenshotCount"] > 0
        assert (output / "index.html").exists()
        html = (output / "index.html").read_text(encoding="utf-8")
        assert "Content-Security-Policy" not in html, "raw build must not include Milestone 1.4-style CSP hardening"
        assert "product/debug" not in html.lower() and "epoch" not in html.lower(), "raw build must not include polished-freeze mode toggles"

        metrics_too_short = build_walkthrough(output, "a.jpg", "b.jpg", hero_a_layers, hero_b_layers, shell, duration_s=1.0)
        assert metrics_too_short["durationSeconds"] >= MIN_DURATION_S


def test_sha256_bytes_matches_hashlib() -> None:
    data = b"remember-milestone-1.5"
    assert sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def main() -> None:
    tests = [
        test_hero_pair_selection,
        test_output_safety,
        test_depth_layers_preserve_photo_fidelity,
        test_depth_layers_reject_no_projected_evidence,
        test_connective_shell_is_subordinate_and_imagined,
        test_connective_shell_learned_runner_seam,
        test_composite_offset_frame_shifts_layers,
        test_walkthrough_duration_clamped_and_raw,
        test_sha256_bytes_matches_hashlib,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} Milestone 1.5 synthetic tests passed.")


if __name__ == "__main__":
    main()
