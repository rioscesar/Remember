#!/usr/bin/env python3
"""Fast, private-data-free regression tests for the Milestone 0.9 helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, INFERRED, OBSERVED, RECONSTRUCTED
from evidence_doctrine import IMAGINED, apply_provenance_priority, generation_masks, provenance_percentages
from milestone09_walkthrough import (
    aggregate_face_provenance,
    build_object_cards,
    build_missing_face_imaginations,
    canonical_orientation,
    classify_face,
    cluster_residual_points,
    fit_room_envelope,
    splat_atlas,
)


def make_synthetic_room(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, list]:
    """A box-like synthetic room: one dense front wall, a sparser floor, and
    a compact "furniture" blob left over as residual (non-planar) evidence.
    Procedural only -- no real or private photograph is used anywhere here.
    """
    wall = np.column_stack((rng.uniform(-3, 3, 2200), rng.uniform(0, 2.6, 2200), rng.normal(0, .01, 2200)))
    floor = np.column_stack((rng.uniform(-3, 3, 400), rng.normal(0, .01, 400), rng.uniform(0.4, 3.5, 400)))
    blob = np.column_stack((rng.normal(1.2, 0.15, 120), rng.normal(0.6, 0.1, 120), rng.normal(1.3, 0.15, 120)))
    points = np.vstack((wall, floor, blob))
    colors = np.clip(rng.normal(150, 35, (len(points), 3)), 0, 255).astype(np.uint8)
    supports = (
        [{0, 1} if i < 2200 else set() for i in range(2200)]
        + [{0} for _ in range(400)]
        + [{0, 1} for _ in range(120)]
    )
    return points, colors, supports


def main() -> None:
    rng = np.random.default_rng(9)
    points, colors, supports = make_synthetic_room(rng)
    cameras = np.array([[-2, 1.3, -2], [2, 1.3, -2]], dtype=float)

    orientation = canonical_orientation(points, cameras)
    assert orientation["gravityEstimated"] is False
    assert np.isfinite(np.asarray(orientation["rotationWorldToRoom"])).all()

    envelope = fit_room_envelope(points, orientation)
    assert all(float(span) > 0 for span in envelope["spans"])

    # Wall-only splat atlas: verify Doctrine v2 still holds for a face card.
    wall_points, wall_colors, wall_supports = points[:2200], colors[:2200], supports[:2200]
    wall_image, wall_provenance, wall_info = splat_atlas(wall_points, wall_colors, wall_supports, width=160)
    assert wall_info["observedOrReconstructedPercent"] > 0
    assert abs(sum(wall_info[name] for name in (
        "observedPercent", "reconstructedPercent", "inferredPercent", "imaginedPercent", "absentPercent"
    )) - 100) < 1e-6

    # Milestone 1.0 provenance priority: no OBSERVED/RECONSTRUCTED texel may be
    # modified by a lower-priority INFERRED/IMAGINED candidate.
    candidate = np.full_like(wall_image, 255)
    candidate_provenance = np.full_like(wall_provenance, IMAGINED)
    locked = np.isin(wall_provenance, [OBSERVED, RECONSTRUCTED])
    protected_image, protected_provenance = apply_provenance_priority(
        wall_image, wall_provenance, candidate, candidate_provenance, locked
    )
    assert np.array_equal(protected_image[locked], wall_image[locked])
    assert np.array_equal(protected_provenance[locked], wall_provenance[locked])

    # Critical unknowns are LOCKED/ABSENT, not GENERATABLE.
    toy_provenance = np.full((8, 8), ABSENT, dtype=np.uint8)
    toy_structural = np.ones((8, 8), dtype=bool)
    toy_critical = np.zeros((8, 8), dtype=bool)
    toy_critical[2:4, 2:4] = True
    toy_masks = generation_masks(toy_provenance, toy_structural, toy_critical)
    assert int((toy_masks["generatable"] & toy_critical).sum()) == 0
    assert int((toy_masks["locked"] & toy_critical).sum()) == int(toy_critical.sum())
    assert provenance_percentages(toy_provenance)["absentPercent"] == 100

    # classify_face: the wall's normal is ~(0,0,1) in world space and should
    # resolve to the room's dominant depth-axis face (front or back).
    wall_normal = np.array([0.0, 0.0, 1.0])
    face_key = classify_face(wall_normal, wall_points, orientation)
    assert face_key in ("front", "back"), face_key

    # Residual clustering: the compact "furniture" blob (last 120 points)
    # must survive as at least one cluster with a real measured position.
    residual = {"points": points[2600:], "colors": colors[2600:], "supports": supports[2600:]}
    clusters = cluster_residual_points(residual["points"], min_size=30)
    assert len(clusters) >= 1, "expected the synthetic furniture blob to survive clustering"
    cards, rejected = build_object_cards(residual, orientation, envelope)
    assert rejected == 0
    assert len(cards) >= 1
    assert all(card["pointCount"] > 0 for card in cards)
    assert all(all(np.isfinite(card["localCenter"])) for card in cards)
    assert all("no shape completion" in card["label"] for card in cards)

    # Room-spanning artifact rejection: a "residual" set that is really just
    # the whole diffuse wall+floor cloud (no compact object at all) must be
    # rejected as a clustering artifact, not fabricated into an object card.
    whole_scene_residual = {"points": points, "colors": colors, "supports": supports}
    whole_scene_cards, whole_scene_rejected = build_object_cards(whole_scene_residual, orientation, envelope)
    assert whole_scene_rejected >= 1, "expected the room-spanning cloud to be rejected, not placed as an object"

    # Missing room faces are generated first in canonical face/atlas space,
    # while recovered face textures stay untouched.
    faces = {
        key: {"recovered": False, "widthPx": 160.0, "heightPx": 100.0, "transform": f"synthetic-{key}",
              "image": None, "provenance": None}
        for key in ("left", "right", "floor", "ceiling", "back", "front")
    }
    faces["front"]["recovered"] = True
    faces["front"]["image"] = wall_image.copy()
    faces["front"]["provenance"] = wall_provenance.copy()
    reports = {key: {"recovered": faces[key]["recovered"]} for key in faces}
    generated_faces = build_missing_face_imaginations(faces, reports, atlas_width=64)
    assert len(generated_faces) == 5
    assert np.array_equal(faces["front"]["image"], wall_image)
    assert np.array_equal(faces["front"]["provenance"], wall_provenance)
    assert all(np.any(faces[key]["imaginedProvenance"] == IMAGINED) for key in generated_faces)
    aggregate = aggregate_face_provenance(faces)
    assert abs(sum(aggregate[name] for name in (
        "observedPercent", "reconstructedPercent", "inferredPercent", "imaginedPercent", "absentPercent"
    )) - 100) < 1e-6
    assert all(faces[key]["transform"].startswith("synthetic-") for key in faces)

    result = {
        "test": "milestone09_synthetic",
        "passed": True,
        "orientationFinite": True,
        "wallObservedOrReconstructedPercent": wall_info["observedOrReconstructedPercent"],
        "wallProvenanceHasReconstructed": bool(np.any(wall_provenance == RECONSTRUCTED)),
        "wallFaceClassifiedAs": face_key,
        "zeroObservedReconstructedModification": True,
        "zeroCriticalOverwrite": True,
        "generatedFaces": len(generated_faces),
        "generatedObjectGaps": 0,
        "aggregateProvenancePercentages": aggregate,
        "spatialConsistencyTransformsPreserved": True,
        "latencyVramMetricsRequiredByProduction": ["latencyMs", "vramBeforeMb", "vramAfterMb", "vramDeltaMb"],
        "residualClusterCount": len(clusters),
        "objectCardCount": len(cards),
        "roomSpanningArtifactsRejected": whole_scene_rejected,
        "wholeSceneResidualAcceptedCompactCards": len(whole_scene_cards),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()