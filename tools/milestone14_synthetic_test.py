#!/usr/bin/env python3
"""Private-data-free contract tests for the Milestone 1.4 demo freeze."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from milestone14_demo_freeze import (
    PATH_DURATION_SECONDS,
    build_manifest,
    render_site,
    require_exact_asset,
    require_private_output,
    validate_provenance,
)


def test_hash_gate() -> None:
    with tempfile.TemporaryDirectory() as temp:
        asset = Path(temp) / "asset.bin"
        asset.write_bytes(b"accepted synthetic fixture")
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        assert require_exact_asset(asset, "back", digest) == digest
        try:
            require_exact_asset(asset, "back", "0" * 64)
            raise AssertionError("changed asset must fail closed")
        except ValueError:
            pass


def test_private_output_gate() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp) / "repo"
        root.mkdir()
        try:
            require_private_output(root / "demo", root)
            raise AssertionError("output inside Git must be rejected")
        except ValueError:
            pass
        require_private_output(Path(temp) / "private-demo", root)


def test_exact_provenance_gate() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        m13 = root / "m13.json"
        m13a = root / "m13a.json"
        m13.write_text(json.dumps({
            "protection": {
                "criticalRegionFace": "front",
                "criticalRegionLabel": "painting",
                "criticalRegionDetectedFromCapturedEvidence": True,
                "protectedModifications": 0,
                "criticalViolations": 0,
                "noFabricatedPerson": True,
            },
            "continuity": {"seamChecksTotal": 2, "seamChecksPassed": 2},
        }))
        m13a.write_text(json.dumps({
            "back_candidates": [{
                "accepted": True, "refinementStrength": .35, "seed": 1301,
                "protectedModifiedPixels": 0, "criticalViolations": 0,
                "consistencyPassed": True,
            }],
            "right_candidates": [{
                "accepted": True, "refinementStrength": .35, "seed": 1101,
                "protectedModifiedPixels": 0, "criticalViolations": 0,
                "consistencyPassed": True,
            }],
        }))
        result = validate_provenance(m13, m13a)
        assert result["criticalTruth"]["class"] == "painting"
        assert result["selected"]["back"]["seed"] == 1301
        assert result["selected"]["right"]["seed"] == 1101
        changed = json.loads(m13a.read_text())
        changed["right_candidates"][0]["seed"] = 1301
        m13a.write_text(json.dumps(changed))
        try:
            validate_provenance(m13, m13a)
            raise AssertionError("wrong accepted seed must fail closed")
        except ValueError:
            pass


def test_offline_presentation_contract() -> None:
    provenance = {
        "selected": {},
        "criticalTruth": {
            "face": "front", "class": "painting",
            "detectedFromCapturedEvidence": True,
            "protectedModifications": 0, "criticalViolations": 0,
        },
        "seams": {"checksPassed": 2, "checksTotal": 2},
    }
    manifest = build_manifest(provenance, {"front": "a", "back": "b", "right": "c"})
    html = render_site(manifest)
    assert manifest["path"]["stopCount"] == 5
    assert manifest["path"]["durationSeconds"] == PATH_DURATION_SECONDS
    assert html.count('id:"') == 5
    for required in (
        "Remember", "Imagine", "Reset", "Play path", "Provenance",
        "The painting stays exactly itself.", "transition:opacity 900ms",
        "myEpoch!==epoch", "Content-Security-Policy",
    ):
        assert required in html
    for forbidden in ("http://", "https://", "fetch(", "XMLHttpRequest", "WebSocket"):
        assert forbidden not in html
    assert "window.addEventListener(\"keydown\"" in html
    assert "window.addEventListener(\"resize\"" not in html
    assert "object-fit:cover" in html
    assert "myEpoch!==epoch" in html


def main() -> None:
    tests = [
        test_hash_gate,
        test_private_output_gate,
        test_exact_provenance_gate,
        test_offline_presentation_contract,
    ]
    for test in tests:
        test()
    print(json.dumps({
        "test": "milestone14_synthetic",
        "usesRealOrPrivatePhotos": False,
        "testsPassed": len(tests),
        "passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
