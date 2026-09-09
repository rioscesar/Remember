#!/usr/bin/env python3
"""Milestone 0.8, Phase 9: human-face guardrail test.

Proves that the actual production inference routine
(`evidence_doctrine.bounded_inference_fill`) never completes an
identity-critical region -- using a purely synthetic, procedurally drawn
stand-in image (ellipses/lines via OpenCV primitives). No real or private
photograph, and no AI-generated imagery, is used for this test, per the
founder brief.

The synthetic image is split into:
  - a "background" region that is always supported (structural, fully
    OBSERVED),
  - a "face" region (an oval + simple features) that is marked BOTH
    unsupported AND critical -- exactly the situation a partially
    photographed real face falling outside every camera's view would
    produce.

If `bounded_inference_fill` ever paints inside the face region, this test
fails loudly. It must import and call the real function -- not a
reimplementation -- so the guardrail proves the production code path, not
a stand-in for it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, INFERRED, bounded_inference_fill


def build_synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (color_bgr, supported_mask, critical_mask) for a synthetic
    room-like backdrop with one unsupported, critical "face" region."""
    height, width = 480, 640
    color = np.full((height, width, 3), (200, 190, 170), dtype=np.uint8)  # flat "wall" backdrop
    cv2.rectangle(color, (0, height - 80), (width, height), (90, 120, 140), -1)  # "floor" strip

    # A procedurally drawn stand-in face -- NOT a real photo, NOT AI-generated.
    face_center = (width // 2, height // 2 - 20)
    cv2.ellipse(color, face_center, (70, 90), 0, 0, 360, (170, 200, 220), -1)
    cv2.ellipse(color, (face_center[0] - 25, face_center[1] - 15), (10, 6), 0, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(color, (face_center[0] + 25, face_center[1] - 15), (10, 6), 0, 0, 360, (60, 60, 60), -1)
    cv2.ellipse(color, (face_center[0], face_center[1] + 30), (25, 12), 0, 0, 180, (60, 60, 90), 3)

    critical_mask = np.zeros((height, width), dtype=bool)
    cv2.ellipse(
        critical_mask.view(np.uint8), face_center, (85, 105), 0, 0, 360, 1, -1
    )
    critical_mask = critical_mask.view(bool)

    # Unsupported region: the face plus a ring around it (simulating a photo
    # boundary that clipped the face and its immediate surroundings), while
    # the rest of the scene (wall/floor backdrop) is fully supported.
    unsupported_mask = np.zeros((height, width), dtype=bool)
    cv2.ellipse(unsupported_mask.view(np.uint8), face_center, (110, 130), 0, 0, 360, 1, -1)
    unsupported_mask = unsupported_mask.view(bool)
    supported_mask = ~unsupported_mask

    # Unsupported pixels carry no real observation at all -- matching how an
    # atlas texel with zero contributing views looks in structural_completion.py
    # (colour 0). Painting a full face into "unsupported" colour data would
    # misrepresent what an unphotographed region actually looks like.
    color[unsupported_mask] = 0

    return color, supported_mask, critical_mask


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("face-guardrail-test-output")
    output.mkdir(parents=True, exist_ok=True)

    color, supported, critical = build_synthetic_scene()
    filled, provenance_delta = bounded_inference_fill(
        color, supported, critical, max_distance_px=200.0
    )

    critical_ever_filled = int((critical & (provenance_delta == INFERRED)).sum())
    critical_pixel_count = int(critical.sum())
    unsupported_noncritical = (~supported) & (~critical)
    noncritical_filled = int((unsupported_noncritical & (provenance_delta == INFERRED)).sum())
    critical_remains_absent = int((critical & (provenance_delta == ABSENT)).sum())

    passed = critical_ever_filled == 0 and critical_remains_absent == critical_pixel_count

    result = {
        "test": "face_guardrail_test",
        "usesRealOrPrivatePhotos": False,
        "usesGeneratedImagery": False,
        "criticalRegionPixelCount": critical_pixel_count,
        "criticalPixelsEverInferredOver": critical_ever_filled,
        "criticalPixelsRemainingAbsent": critical_remains_absent,
        "nonCriticalUnsupportedPixelsInferred": noncritical_filled,
        "passed": passed,
    }

    before_vis = color.copy()
    before_vis[~supported] = (before_vis[~supported] * 0.3).astype(np.uint8)
    before_vis[critical] = (0, 0, 220)

    cv2.imwrite(str(output / "synthetic-before.png"), before_vis)
    cv2.imwrite(str(output / "synthetic-after-fill.png"), filled)
    marked_after = filled.copy()
    marked_after[critical] = (
        0.5 * marked_after[critical] + 0.5 * np.array([0, 0, 220])
    ).astype(np.uint8)
    cv2.imwrite(str(output / "synthetic-after-fill-critical-outline.png"), marked_after)
    (output / "face-guardrail-metrics.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("FACE GUARDRAIL TEST FAILED: critical region was inferred over.")


if __name__ == "__main__":
    main()
