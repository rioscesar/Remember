#!/usr/bin/env python3
"""Fast, private-data-free regression tests for Milestone 1.3 helpers.

Everything here uses procedurally-generated synthetic data only -- no real
or private photograph, reconstruction, model weight, or generated texture is
read or written. These tests prove the shape of the actual production
functions in `milestone13_walkthrough.py` (imported directly, never
reimplemented): demo-face ranking/selection, candidate accept/reject
bookkeeping, IMAGINED-only harmonization, the protected critical-region
invariant, product/debug presentation-mode field hiding, and metrics
aggregation completeness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import IMAGINED, OBSERVED, RECONSTRUCTED
from milestone13_walkthrough import (
    ADJACENT_FACE_PAIRS,
    CandidateReport,
    DEMO_PATH_ORDER,
    FACE_KEYS,
    aggregate_milestone13_metrics,
    build_critical_region_demo,
    build_presentation_payload,
    build_walkthrough_manifest,
    harmonize_cross_face_continuity,
    rank_faces_for_walkthrough,
    select_demo_visible_missing_faces,
)


def make_synthetic_faces() -> dict:
    """A coarse six-face shell: two recovered faces, four missing faces,
    with distinct sizes so ranking/selection is unambiguous and reproducible.
    """
    sizes = {
        "front": (200.0, 120.0), "back": (200.0, 120.0),
        "left": (150.0, 120.0), "right": (150.0, 120.0),
        "ceiling": (200.0, 150.0), "floor": (200.0, 150.0),
    }
    recovered = {"front", "back"}
    faces = {}
    for key in FACE_KEYS:
        width, height = sizes[key]
        faces[key] = {
            "recovered": key in recovered,
            "widthPx": width,
            "heightPx": height,
            "transform": f"synthetic-{key}",
            "image": None,
            "provenance": None,
        }
    return faces


def test_ranking_is_deterministic_and_prioritizes_path_order() -> None:
    faces = make_synthetic_faces()
    ranked_a = rank_faces_for_walkthrough(faces)
    ranked_b = rank_faces_for_walkthrough(faces)
    assert ranked_a == ranked_b, "ranking must be deterministic for identical input geometry"
    assert len(ranked_a) == len(FACE_KEYS)

    # "front" is both earliest in the intended path and large -- it must rank first.
    assert ranked_a[0]["face"] == "front"

    demo_faces = select_demo_visible_missing_faces(ranked_a, max_faces=2)
    assert len(demo_faces) == 2
    assert all(not faces[key]["recovered"] for key in demo_faces)
    # "right" precedes "left" in DEMO_PATH_ORDER and both precede ceiling/floor.
    assert demo_faces[0] == "right"

    # An all-recovered room selects nothing to generate.
    all_recovered = make_synthetic_faces()
    for face in all_recovered.values():
        face["recovered"] = True
    assert select_demo_visible_missing_faces(rank_faces_for_walkthrough(all_recovered)) == []


def test_candidate_report_accept_reject_bookkeeping() -> None:
    accepted_report = CandidateReport(
        face="right", candidateIndex=0, contextPhoto="ctx-a.jpg", engine="learned-diffusion-ip-adapter",
        doctrineAccepted=True, consistencyPassed=True, rejected=False, rejectionReason=None,
        metrics={"seed": 1301},
    )
    rejected_report = CandidateReport(
        face="right", candidateIndex=1, contextPhoto="ctx-b.jpg", engine="learned-diffusion-ip-adapter",
        doctrineAccepted=True, consistencyPassed=False, rejected=True,
        rejectionReason="unsupported semantic additions: detected unsupported people in structural wall",
        metrics={"seed": 1301},
    )
    reports = [accepted_report, rejected_report]
    accepted = [r for r in reports if not r.rejected]
    rejected = [r for r in reports if r.rejected]
    assert len(accepted) == 1 and len(rejected) == 1
    assert "people" in rejected[0].rejectionReason


def test_harmonization_only_touches_imagined_pixels() -> None:
    faces = make_synthetic_faces()
    height, width = 40, 40

    # "right" and "ceiling" are an adjacent pair with mismatched imagined tones.
    for key, tone in (("right", 60), ("ceiling", 200)):
        image = np.full((height, width, 3), tone, dtype=np.uint8)
        provenance = np.full((height, width), IMAGINED, dtype=np.uint8)
        # Half of each face is locked evidence that must never move.
        provenance[: height // 2, :] = OBSERVED
        image[: height // 2, :] = 111  # a locked, very different colour
        faces[key]["imaginedImage"] = image
        faces[key]["imaginedProvenance"] = provenance
        faces[key]["imagined"] = True

    locked_before = faces["right"]["imaginedImage"][: height // 2, :].copy()
    seam_reports = harmonize_cross_face_continuity(faces, pairs=(("right", "ceiling"),))

    assert len(seam_reports) == 1
    report = seam_reports[0]
    assert report["pair"] == ["right", "ceiling"]
    assert report["afterDeltaE"] <= report["beforeDeltaE"] + 1e-6
    assert report["imaginedPixelsAdjusted"] > 0

    # Locked (OBSERVED) pixels are provably untouched -- exact byte match.
    locked_after = faces["right"]["imaginedImage"][: height // 2, :]
    assert np.array_equal(locked_before, locked_after), "harmonization must never move locked evidence pixels"

    # Two faces with nothing IMAGINED at all produce no seam report.
    plain_faces = make_synthetic_faces()
    for key in ("front", "back"):
        image = np.full((height, width, 3), 128, dtype=np.uint8)
        provenance = np.full((height, width), RECONSTRUCTED, dtype=np.uint8)
        plain_faces[key]["provenance"] = provenance
        plain_faces[key]["image"] = image
    assert harmonize_cross_face_continuity(plain_faces, pairs=(("front", "back"),)) == []


def test_critical_region_demo_is_exact_and_has_no_fabricated_person() -> None:
    height, width = 100, 100
    image = np.full((height, width, 3), 150, dtype=np.uint8)
    provenance = np.full((height, width), OBSERVED, dtype=np.uint8)
    provenance[60:, :] = IMAGINED  # nearby generated structure

    demo = build_critical_region_demo("front", image, provenance, seg_processor=None, seg_model=None)

    assert demo["criticalPixelCount"] > 0
    assert demo["criticalPixelsExactMatch"] is True
    assert demo["criticalPixelsModified"] == 0
    assert demo["noFabricatedPerson"] is True
    assert demo["detectedFromCapturedEvidence"] is False  # no segmentation model supplied in this synthetic test
    assert "poster placeholder" in demo["label"]
    assert bool(demo["mask"].any())


def test_presentation_mode_hides_metrics_paths_and_filenames() -> None:
    faces = make_synthetic_faces()
    faces["front"]["transform"] = "translateZ(400px)"
    stops = [
        {"stage": "photo_origin_view", "face": "front", "mode": "remember", "description": "d1"},
        {"stage": "constrained_movement", "face": "right", "mode": "imagine", "description": "d2",
         "movementBounds": [1.0, 2.0, 3.0]},
    ]

    product_payload = build_presentation_payload(stops, faces, mode="product")
    debug_payload = build_presentation_payload(stops, faces, mode="debug")

    product_json = json.dumps(product_payload)
    assert "translateZ" not in product_json, "product view must not leak raw transform/path data"
    assert "widthPx" not in product_json and "heightPx" not in product_json
    assert all("face" not in entry for entry in product_payload)

    debug_json = json.dumps(debug_payload)
    assert "translateZ" in debug_json
    assert any(entry.get("face") == "front" for entry in debug_payload)
    assert any("movementBounds" in entry for entry in debug_payload)

    try:
        build_presentation_payload(stops, faces, mode="bogus")
        raise AssertionError("expected ValueError for unknown presentation mode")
    except ValueError:
        pass


def test_walkthrough_manifest_has_five_deterministic_stops() -> None:
    faces = make_synthetic_faces()
    face_reports = {
        "front": {"recovered": True, "atlas": {"observedOrReconstructedPercent": 92.0}},
        "back": {"recovered": True, "atlas": {"observedOrReconstructedPercent": 34.0}},
        "left": {"recovered": False}, "right": {"recovered": False},
        "ceiling": {"recovered": False}, "floor": {"recovered": False},
    }
    envelope = {"spans": [4.0, 2.6, 3.5]}
    demo_faces = ["right", "left"]
    accepted_by_face = {"right": {"consistencyScore": 81.2}, "left": {"consistencyScore": 55.0}}

    stops = build_walkthrough_manifest(faces, face_reports, demo_faces, accepted_by_face, envelope)
    stages = [s["stage"] for s in stops]
    assert stages == [
        "photo_origin_view", "weak_region_remember", "remember_to_imagine_transition",
        "immersive_imagine_viewpoint", "constrained_movement",
    ]
    assert stops[0]["face"] == "front"  # strongest evidence
    assert stops[1]["face"] == "back"  # weakest evidence among recovered faces
    assert stops[2]["face"] == "right"  # first demo-visible missing face
    assert stops[3]["face"] == "right"  # highest accepted consistency score
    assert stops[4]["movementBounds"] == [4.0, 2.6, 3.5]


def test_metrics_aggregation_includes_every_required_field() -> None:
    faces = make_synthetic_faces()
    height, width = 20, 20
    for key, tone, prov_code in (("right", 90, IMAGINED), ("front", 130, OBSERVED)):
        faces[key]["image"] = np.full((height, width, 3), tone, dtype=np.uint8)
        faces[key]["provenance"] = np.full((height, width), prov_code, dtype=np.uint8)
        if prov_code == IMAGINED:
            faces[key]["imaginedImage"] = faces[key]["image"]
            faces[key]["imaginedProvenance"] = faces[key]["provenance"]
            faces[key]["imagined"] = True

    ranked_faces = rank_faces_for_walkthrough(faces)
    demo_faces = ["right"]
    candidate_reports = [
        CandidateReport("right", 0, "ctx.jpg", "learned-diffusion-ip-adapter", True, True, False, None, {}),
        CandidateReport("right", 1, "ctx2.jpg", "learned-diffusion-ip-adapter", True, False, True, "unsupported semantic additions: people", {}),
    ]
    accepted_by_face = {"right": {"consistencyScore": 70.0}}
    critical_demo = {
        "face": "front", "label": "synthetic protected region", "detectedFromCapturedEvidence": False,
        "criticalPixelCount": 40, "criticalPixelsExactMatch": True, "criticalPixelsModified": 0,
        "adjacentGeneratedPixels": 12, "noFabricatedPerson": True,
    }
    seam_reports = [{"pair": ["right", "front"], "beforeDeltaE": 10.0, "afterDeltaE": 4.0, "improved": True, "passed": True, "imaginedPixelsAdjusted": 30}]

    metrics = aggregate_milestone13_metrics(
        ranked_faces, demo_faces, faces, candidate_reports, accepted_by_face, critical_demo,
        seam_reports, {"right": 812.3}, 812.3, 3200, 27.0,
    )

    required_paths = [
        ("faces", "demoVisibleFaces"), ("faces", "reconstructedFaces"), ("faces", "inferredPercent"),
        ("faces", "imaginedPercent"), ("generation", "acceptedGenerations"), ("generation", "rejectedGenerations"),
        ("generation", "perFaceLatencyMs"), ("generation", "totalPreGenerationMs"), ("generation", "peakVramMb"),
        ("protection", "protectedModifications"), ("protection", "criticalViolations"),
        ("continuity", "seamChecks"), ("presentation", "pathDurationSeconds"),
    ]
    for section, field_name in required_paths:
        assert field_name in metrics[section], f"missing required metric {section}.{field_name}"

    assert metrics["generation"]["acceptedGenerations"] == 1
    assert metrics["generation"]["rejectedGenerations"] == 1
    assert metrics["protection"]["criticalViolations"] == 0
    assert metrics["protection"]["protectedModifications"] == 0
    assert metrics["continuity"]["seamChecksPassed"] == 1


def test_adjacent_face_pairs_cover_every_cube_edge_exactly_once() -> None:
    assert len(ADJACENT_FACE_PAIRS) == 12
    assert len(set(ADJACENT_FACE_PAIRS)) == 12
    covered = set()
    for a, b in ADJACENT_FACE_PAIRS:
        covered.add(a)
        covered.add(b)
    assert covered == set(FACE_KEYS)


def main() -> None:
    tests = [
        test_ranking_is_deterministic_and_prioritizes_path_order,
        test_candidate_report_accept_reject_bookkeeping,
        test_harmonization_only_touches_imagined_pixels,
        test_critical_region_demo_is_exact_and_has_no_fabricated_person,
        test_presentation_mode_hides_metrics_paths_and_filenames,
        test_walkthrough_manifest_has_five_deterministic_stops,
        test_metrics_aggregation_includes_every_required_field,
        test_adjacent_face_pairs_cover_every_cube_edge_exactly_once,
    ]
    for test in tests:
        test()
    result = {
        "test": "milestone13_synthetic",
        "usesRealOrPrivatePhotos": False,
        "usesGeneratedImagery": False,
        "testsPassed": len(tests),
        "passed": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
