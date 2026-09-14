#!/usr/bin/env python3
"""Fast, private-data-free regression tests for the Milestone 0.9 helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import ABSENT, INFERRED, OBSERVED, RECONSTRUCTED
from milestone09_walkthrough import canonical_orientation, evidence_texture, fit_room_envelope


def main() -> None:
    rng = np.random.default_rng(9)
    # A box-like room with a visibly dominant wall and two camera positions.
    wall = np.column_stack((rng.uniform(-3, 3, 1800), rng.uniform(0, 2.6, 1800), rng.normal(0, .01, 1800)))
    points = np.vstack((wall, np.column_stack((rng.uniform(-3, 3, 300), rng.uniform(0, 2.6, 300), rng.uniform(2, 4, 300)))))
    colors = np.clip(rng.normal(150, 35, (len(points), 3)), 0, 255).astype(np.uint8)
    supports = [{0, 1} if i < 1800 else {0} for i in range(len(points))]
    cameras = np.array([[-2, 1.3, 2], [2, 1.3, 2]], dtype=float)
    orientation = canonical_orientation(points, cameras)
    envelope = fit_room_envelope(points, orientation)
    assert all(float(span) > 0 for span in envelope["spans"])
    critical = np.zeros((80, 160), dtype=bool)
    critical[30:50, 70:90] = True
    _, provenance = evidence_texture(points, colors, supports, orientation, 160, 80, critical)
    assert not np.any(provenance[critical] == INFERRED)
    assert np.count_nonzero(provenance == RECONSTRUCTED) > 0
    result = {"test": "milestone09_synthetic", "passed": True,
              "criticalPixelsInferred": int(np.count_nonzero(provenance[critical] == INFERRED)),
              "reconstructedPixels": int(np.count_nonzero(provenance == RECONSTRUCTED)),
              "orientationFinite": bool(np.isfinite(np.asarray(orientation["rotationWorldToRoom"])).all())}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
