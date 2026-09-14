#!/usr/bin/env python3
"""Synthetic guardrail tests for optional learned atlas inpainting."""

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


def fake_runner(image_rgb: np.ndarray, mask: np.ndarray, prompt: str) -> np.ndarray:
    generated = image_rgb.copy()
    generated[mask > 0] = np.array([181, 176, 164], dtype=np.uint8)
    generated[mask == 0] = np.array([255, 0, 255], dtype=np.uint8)
    if "boring" not in prompt or "no people" not in prompt:
        raise RuntimeError("prompt is not the expected structural prompt")
    return generated


def main() -> None:
    base, provenance, generatable, locked, critical = build_scene()
    result = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        face_key="back",
        config=LearnedInpaintingConfig(max_resolution=128, steps=2),
        runner=fake_runner,
    )
    if not result.accepted or result.image_bgr is None or result.provenance is None:
        raise SystemExit(f"expected learned synthetic runner to be accepted: {result.blocker}")
    protected = ~generatable
    protected_modified = int(np.any(result.image_bgr[protected] != base[protected], axis=1).sum())
    critical_violations = int(((result.provenance == IMAGINED) & critical).sum())
    imagined_generatable = int(((result.provenance == IMAGINED) & generatable).sum())
    if protected_modified != 0 or critical_violations != 0 or imagined_generatable != int(generatable.sum()):
        raise SystemExit("mask/provenance enforcement failed")

    fail_closed = run_learned_inpainting(
        base,
        provenance,
        generatable,
        locked,
        critical,
        config=LearnedInpaintingConfig(model_id="remember/nonexistent-local-model", allow_model_download=False),
    )
    if fail_closed.accepted or not fail_closed.blocker or not fail_closed.metrics.get("failClosed"):
        raise SystemExit("missing local model/dependencies did not fail closed")

    output = {
        "test": "learned_inpainting_test",
        "usesRealOrPrivatePhotos": False,
        "usesCloudProcessing": False,
        "dependenciesPresent": learned_dependencies_status(),
        "protectedModifiedPixels": protected_modified,
        "criticalViolations": critical_violations,
        "imaginedGeneratablePixels": imagined_generatable,
        "failClosedBlocker": fail_closed.blocker,
        "passed": True,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
