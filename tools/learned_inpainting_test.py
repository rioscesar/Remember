#!/usr/bin/env python3
"""Synthetic guardrail and contextual provenance tests for Remember Milestone 1.2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, IMAGINED, OBSERVED, RECONSTRUCTED
from learned_inpainting import (
    LearnedInpaintingConfig,
    learned_dependencies_status,
    run_learned_inpainting,
)
from context_consistency import (
    check_palette_consistency,
    check_structural_edge_continuity,
    check_semantic_additions,
    evaluate_context_consistency,
)


def build_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 128, 160
    y, x = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = 80 + (x % 30)
    base[:, :, 1] = 95 + (y % 20)
    base[:, :, 2] = 120

    provenance = np.full((height, width), ABSENT, dtype=np.uint8)
    provenance[:, :40] = OBSERVED
    provenance[:, 40:72] = RECONSTRUCTED
    base[:, 72:] = 0

    structural = np.ones((height, width), dtype=bool)
    critical = np.zeros((height, width), dtype=bool)
    cv2.rectangle(critical.view(np.uint8), (100, 32), (135, 84), 1, -1)
    critical = critical.view(bool)
    locked = np.isin(provenance, [OBSERVED, RECONSTRUCTED]) | critical
    generatable = (provenance == ABSENT) & structural & ~locked
    return base, provenance, generatable, locked, critical


def fake_runner_unconditioned(image_rgb: np.ndarray, mask: np.ndarray, prompt: str, context_rgb: np.ndarray | None = None) -> np.ndarray:
    generated = image_rgb.copy()
    generated[mask > 0] = np.array([181, 176, 164], dtype=np.uint8)
    generated[mask == 0] = np.array([255, 0, 255], dtype=np.uint8)
    if "boring" not in prompt or "no people" not in prompt:
        raise RuntimeError("prompt is not the expected structural prompt")
    return generated


def fake_runner_conditioned(image_rgb: np.ndarray, mask: np.ndarray, prompt: str, context_rgb: np.ndarray | None = None) -> np.ndarray:
    if context_rgb is None:
        raise RuntimeError("expected context_rgb for conditioned runner")
    generated = image_rgb.copy()
    # Apply reference color from context
    mean_ref = context_rgb.mean(axis=(0, 1)).astype(np.uint8)
    generated[mask > 0] = mean_ref
    generated[mask == 0] = np.array([255, 0, 255], dtype=np.uint8)
    return generated


def test_unconditioned_guardrails() -> dict:
    base, provenance, generatable, locked, critical = build_scene()
    result = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        face_key="back",
        config=LearnedInpaintingConfig(max_resolution=128, steps=2),
        runner=fake_runner_unconditioned,
    )
    if not result.accepted or result.image_bgr is None or result.provenance is None:
        raise SystemExit(f"expected learned synthetic runner to be accepted: {result.blocker}")
    protected = ~generatable
    protected_modified = int(np.any(result.image_bgr[protected] != base[protected], axis=1).sum())
    critical_violations = int(((result.provenance == IMAGINED) & critical).sum())
    imagined_generatable = int(((result.provenance == IMAGINED) & generatable).sum())
    if protected_modified != 0 or critical_violations != 0 or imagined_generatable != int(generatable.sum()):
        raise SystemExit("unconditioned mask/provenance enforcement failed")
    return {
        "protectedModifiedPixels": protected_modified,
        "criticalViolations": critical_violations,
        "imaginedGeneratablePixels": imagined_generatable,
    }


def test_contextual_provenance_doctrine() -> dict:
    """Validate that contextual evidence informs generation but NEVER gets relabeled as RECONSTRUCTED."""
    base, provenance, generatable, locked, critical = build_scene()
    synthetic_context = np.full((64, 64, 3), (170, 160, 145), dtype=np.uint8)

    config = LearnedInpaintingConfig(
        max_resolution=128,
        steps=2,
        use_ip_adapter=True,
        context_image=synthetic_context,
    )
    result = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        face_key="back",
        config=config,
        runner=fake_runner_conditioned,
    )
    if not result.accepted or result.image_bgr is None or result.provenance is None:
        raise SystemExit(f"expected contextual runner to be accepted: {result.blocker}")

    # Critical check: Contextually conditioned pixels MUST be IMAGINED, never OBSERVED or RECONSTRUCTED
    contextual_provenance = result.provenance[generatable]
    non_imagined_in_generated = int((contextual_provenance != IMAGINED).sum())
    reconstructed_in_generated = int((contextual_provenance == RECONSTRUCTED).sum())
    observed_in_generated = int((contextual_provenance == OBSERVED).sum())

    if non_imagined_in_generated != 0 or reconstructed_in_generated != 0 or observed_in_generated != 0:
        raise SystemExit("FATAL DOCTRINE VIOLATION: contextual evidence was relabeled as RECONSTRUCTED/OBSERVED!")

    protected = ~generatable
    protected_modified = int(np.any(result.image_bgr[protected] != base[protected], axis=1).sum())
    critical_violations = int(((result.provenance == IMAGINED) & critical).sum())

    if protected_modified != 0 or critical_violations != 0:
        raise SystemExit("contextual run violated protected/critical masks")

    return {
        "contextualProvenanceEnforced": True,
        "reconstructedInGenerated": reconstructed_in_generated,
        "observedInGenerated": observed_in_generated,
        "imaginedInGenerated": int((contextual_provenance == IMAGINED).sum()),
        "protectedModifiedPixels": protected_modified,
        "criticalViolations": critical_violations,
    }


def test_context_consistency_checks() -> dict:
    """Test palette Delta E, edge continuity, and semantic additions checks."""
    # 1. Matching palette
    gen_img = np.full((100, 100, 3), (140, 160, 175), dtype=np.uint8)
    ref_img = np.full((100, 100, 3), (138, 158, 173), dtype=np.uint8)
    pal_match = check_palette_consistency(gen_img, ref_img)
    if not pal_match.passed or pal_match.delta_e > 5.0:
        raise SystemExit("palette match test failed")

    # 2. Drifting palette
    ref_drift = np.full((100, 100, 3), (20, 20, 220), dtype=np.uint8)  # bright red
    pal_drift = check_palette_consistency(gen_img, ref_drift)
    if pal_drift.passed or pal_drift.delta_e < 35.0:
        raise SystemExit("palette drift test failed to flag large Delta E")

    # 3. Edge continuity
    edge_res = check_structural_edge_continuity(gen_img)
    if not edge_res.passed:
        raise SystemExit("edge continuity on smooth image failed")

    # 4. Semantic additions heuristic check
    sem_res = check_semantic_additions(gen_img)
    if not sem_res.passed:
        raise SystemExit("semantic additions heuristic check failed")

    return {
        "paletteMatchDeltaE": pal_match.delta_e,
        "paletteMatchPassed": pal_match.passed,
        "paletteDriftDeltaE": pal_drift.delta_e,
        "paletteDriftPassed": pal_drift.passed,
        "edgeContinuityScore": edge_res.score,
        "semanticAdditionsPassed": sem_res.passed,
    }


def test_fail_closed_behavior() -> dict:
    base, provenance, generatable, locked, critical = build_scene()

    # Model not present
    fc_model = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        config=LearnedInpaintingConfig(
            model_id="remember/nonexistent-local-model",
            allow_model_download=False,
        ),
    )
    if fc_model.accepted or not fc_model.blocker or not fc_model.metrics.get("failClosed"):
        raise SystemExit("missing local model did not fail closed")

    # IP adapter not present
    fc_ip = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        config=LearnedInpaintingConfig(
            use_ip_adapter=True,
            ip_adapter_model_id="remember/nonexistent-ip-adapter",
            allow_model_download=False,
            context_image=np.zeros((64, 64, 3), dtype=np.uint8),
        ),
    )
    if fc_ip.accepted or not fc_ip.blocker or not fc_ip.metrics.get("failClosed"):
        raise SystemExit("missing local IP-adapter did not fail closed")

    return {
        "missingModelBlocker": fc_model.blocker,
        "missingIpAdapterBlocker": fc_ip.blocker,
    }


def main() -> None:
    uncond = test_unconditioned_guardrails()
    context_prov = test_contextual_provenance_doctrine()
    consistency = test_context_consistency_checks()
    fail_closed = test_fail_closed_behavior()

    output = {
        "test": "learned_inpainting_test",
        "milestone": "1.2",
        "usesRealOrPrivatePhotos": False,
        "usesCloudProcessing": False,
        "dependenciesPresent": learned_dependencies_status(),
        "unconditionedGuardrails": uncond,
        "contextualProvenanceEnforced": context_prov,
        "contextConsistencyChecks": consistency,
        "failClosedChecks": fail_closed,
        "passed": True,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()