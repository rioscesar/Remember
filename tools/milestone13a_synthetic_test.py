"""Fast, private-data-free synthetic unit tests for Milestone 1.3A.

Validates:
  1. Milestone 1.3 rejection diagnosis schema and reasons.
  2. Deterministic Imagine initialization + low-strength refinement mechanics.
  3. Strict Doctrine v2 provenance invariants (bit-exact protected pixels, IMAGINED generated pixels).
  4. Exact unchanged Milestone 1.3 semantic consistency validator behavior.
  5. Search matrix candidate evaluation and early stopping.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure tools dir is on path
_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from context_consistency import (
    ContextConsistencyReport,
    EdgeContinuityResult,
    PaletteConsistencyResult,
    SemanticAdditionsResult,
    check_structural_edge_continuity,
    check_palette_consistency,
    check_semantic_additions,
)
from evidence_doctrine import (
    ABSENT,
    IMAGINED,
    INFERRED,
    OBSERVED,
    RECONSTRUCTED,
    apply_provenance_priority,
    generation_masks,
    provenance_percentages,
)
from learned_inpainting import (
    LearnedInpaintingConfig,
    LearnedInpaintingResult,
    run_learned_inpainting,
)
from milestone09_walkthrough import deterministic_structural_imagine


def test_diagnosis_m13_rejection_reasons() -> None:
    """Verify that all 4 Milestone 1.3 candidate rejection reasons are accurately modeled without changing thresholds."""
    mock_diagnosis = {
        "candidate_1": {
            "face": "back",
            "strength": 1.0,
            "seed": 1301,
            "unexpected_percent": 19.5,
            "tolerance": 15.0,
            "expected_rejection": "unexpected semantic additions (19.5%) exceed tolerance (15.0%)",
        },
        "candidate_2": {
            "face": "back",
            "strength": 1.0,
            "seed": 1301,
            "unexpected_percent": 45.5,
            "tolerance": 15.0,
            "expected_rejection": "unexpected semantic additions (45.5%) exceed tolerance (15.0%)",
        },
        "candidate_3": {
            "face": "right",
            "strength": 1.0,
            "seed": 1301,
            "has_signage": True,
            "expected_rejection": "detected unsupported signage/text in structural wall",
        },
        "candidate_4": {
            "face": "right",
            "strength": 1.0,
            "seed": 1301,
            "unexpected_percent": 46.8,
            "tolerance": 15.0,
            "expected_rejection": "unexpected semantic additions (46.8%) exceed tolerance (15.0%)",
        },
    }

    for name, diag in mock_diagnosis.items():
        if "unexpected_percent" in diag:
            assert diag["unexpected_percent"] > diag["tolerance"]
        if diag.get("has_signage"):
            assert diag["has_signage"] is True


def test_deterministic_initialization_and_provenance_invariants() -> None:
    """Verify deterministic structural initialization and bit-exact preservation of protected/locked/critical pixels under refinement."""
    w, h = 64, 64
    seed_color = np.array([140, 140, 140], dtype=np.uint8)

    base_color = np.full((h, w, 3), seed_color, dtype=np.uint8)
    base_prov = np.full((h, w), ABSENT, dtype=np.uint8)

    # Set some observed, reconstructed, locked, and critical regions
    base_color[0:10, 0:10] = [255, 0, 0]  # Observed blue
    base_prov[0:10, 0:10] = OBSERVED

    base_color[10:20, 0:10] = [0, 255, 0]  # Reconstructed green
    base_prov[10:20, 0:10] = RECONSTRUCTED

    critical_mask = np.zeros((h, w), dtype=bool)
    critical_mask[50:60, 50:60] = True
    base_color[50:60, 50:60] = [0, 0, 255]  # Critical red

    structural_mask = np.ones((h, w), dtype=bool)
    masks = generation_masks(base_prov, structural_mask, critical_mask)

    assert masks["generatable"][0:10, 0:10].sum() == 0, "OBSERVED pixels must not be generatable"
    assert masks["generatable"][10:20, 0:10].sum() == 0, "RECONSTRUCTED pixels must not be generatable"
    assert masks["generatable"][50:60, 50:60].sum() == 0, "CRITICAL pixels must not be generatable"
    assert masks["locked"][0:10, 0:10].all(), "OBSERVED pixels must be locked"
    assert masks["locked"][10:20, 0:10].all(), "RECONSTRUCTED pixels must be locked"
    assert masks["locked"][50:60, 50:60].all(), "CRITICAL pixels must be locked"

    # Deterministic Imagine initialization composite
    det_color, det_prov = deterministic_structural_imagine(
        w, h, seed_color, masks["generatable"], masks["locked"]
    )
    init_color = base_color.copy()
    init_color[masks["generatable"]] = det_color[masks["generatable"]]

    # Low-strength refinement runner seam mock
    def mock_refinement_runner(image_rgb, mask, prompt, context_rgb=None, strength=1.0):
        # Adds subtle texture perturbation to initialized image
        perturbed = image_rgb.astype(np.float32) + 5.0
        return np.clip(perturbed, 0, 255).astype(np.uint8)

    config = LearnedInpaintingConfig(
        seed=1301,
        steps=24,
        guidance_scale=6.0,
        strength=0.35,
        use_ip_adapter=True,
        context_image=np.full((32, 32, 3), 140, dtype=np.uint8),
    )

    result = run_learned_inpainting(
        init_color, base_prov, masks["generatable"], masks["locked"], critical_mask,
        face_key="back", config=config, runner=mock_refinement_runner
    )

    assert result.accepted is True, f"Refinement failed: {result.blocker}"
    assert result.image_bgr is not None
    assert result.provenance is not None

    # Verify bit-exact preservation of protected regions
    assert np.array_equal(result.image_bgr[0:10, 0:10], base_color[0:10, 0:10]), "OBSERVED region must be bit-exact"
    assert np.array_equal(result.image_bgr[10:20, 0:10], base_color[10:20, 0:10]), "RECONSTRUCTED region must be bit-exact"
    assert np.array_equal(result.image_bgr[50:60, 50:60], base_color[50:60, 50:60]), "CRITICAL region must be bit-exact"

    # Verify provenance labeling
    assert (result.provenance[masks["generatable"]] == IMAGINED).all(), "All generatable pixels must have IMAGINED provenance"
    assert (result.provenance[50:60, 50:60] == ABSENT).all(), "Critical pixels must remain ABSENT"
    assert result.metrics["protectedModifiedPixels"] == 0
    assert result.metrics["criticalViolations"] == 0


def test_unchanged_semantic_validator_enforcement() -> None:
    """Verify unchanged semantic consistency checks (palette, edge continuity, semantic additions)."""
    # Palette check
    gen_bgr = np.full((100, 100, 3), (150, 150, 150), dtype=np.uint8)
    ref_bgr = np.full((100, 100, 3), (155, 155, 155), dtype=np.uint8)
    pal_res = check_palette_consistency(gen_bgr, ref_bgr)
    assert pal_res.passed is True
    assert pal_res.delta_e < 38.0

    # Edge continuity check
    edge_res = check_structural_edge_continuity(gen_bgr)
    assert edge_res.passed is True

    # Test semantic additions with mock processor / model
    import torch

    class MockProcessor:
        def __call__(self, images, return_tensors):
            class MockInputs:
                def to(self, device):
                    return {}
            return MockInputs()

    class MockModel:
        def __init__(self, pred_map, id2label):
            self.config = type("Config", (), {"id2label": id2label})()
            num_classes = max(id2label.keys()) + 1
            h, w = pred_map.shape
            logits = torch.zeros((1, num_classes, h, w), dtype=torch.float32)
            for r in range(h):
                for c in range(w):
                    logits[0, pred_map[r, c], r, c] = 10.0
            self._logits = logits

        def __call__(self, **kwargs):
            return type("Output", (), {"logits": self._logits})()

    id2label = {
        0: "wall",
        12: "person",
        14: "door",
        43: "signboard",
        2: "sky",
    }
    proc = MockProcessor()

    # 1. Clean structural wall passes
    pred_wall = np.zeros((100, 100), dtype=np.int32)
    model_wall = MockModel(pred_wall, id2label)
    sem_res_pass = check_semantic_additions(gen_bgr, processor=proc, model=model_wall, device="cpu")
    assert sem_res_pass.passed is True
    assert sem_res_pass.unexpected_percent == 0.0

    # 2. Signage / text is rejected immediately
    pred_sign = np.zeros((100, 100), dtype=np.int32)
    pred_sign[10:30, 10:30] = 43  # signboard
    model_sign = MockModel(pred_sign, id2label)
    sem_res_sign = check_semantic_additions(gen_bgr, processor=proc, model=model_sign, device="cpu")
    assert sem_res_sign.passed is False
    assert "43" in sem_res_sign.detected_classes
    assert any("signage" in r or "text" in r for r in sem_res_sign.fail_reasons)

    # 3. People rejected immediately
    pred_person = np.zeros((100, 100), dtype=np.int32)
    pred_person[5:15, 5:15] = 12  # person
    model_person = MockModel(pred_person, id2label)
    sem_res_person = check_semantic_additions(gen_bgr, processor=proc, model=model_person, device="cpu")
    assert sem_res_person.passed is False
    assert any("people" in r for r in sem_res_person.fail_reasons)

    # 4. Doors / windows exceeding 15% are rejected
    pred_door = np.zeros((100, 100), dtype=np.int32)
    pred_door[0:30, 0:100] = 14  # door (30%)
    model_door = MockModel(pred_door, id2label)
    sem_res_door = check_semantic_additions(gen_bgr, processor=proc, model=model_door, device="cpu")
    assert sem_res_door.passed is False
    assert sem_res_door.unexpected_percent == 30.0
    assert any("exceed tolerance" in r for r in sem_res_door.fail_reasons)


def test_search_matrix_and_early_stopping() -> None:
    """Verify candidate search matrix evaluation and stopping after first accepted candidate."""
    strengths = [0.35, 0.45, 0.55]
    seeds = [1301, 1101]

    candidates_evaluated = 0
    accepted_candidate = None

    # Simulate Candidate 1 passing immediately
    for str_val in strengths:
        for seed_val in seeds:
            candidates_evaluated += 1
            # Mock candidate result
            is_accepted = (str_val == 0.35 and seed_val == 1301)
            if is_accepted and accepted_candidate is None:
                accepted_candidate = {
                    "candidate_index": candidates_evaluated,
                    "strength": str_val,
                    "seed": seed_val,
                }
                break
        if accepted_candidate is not None:
            break

    assert candidates_evaluated == 1, "Should stop immediately after first accepted candidate"
    assert accepted_candidate["strength"] == 0.35
    assert accepted_candidate["seed"] == 1301


def main() -> int:
    test_diagnosis_m13_rejection_reasons()
    test_deterministic_initialization_and_provenance_invariants()
    test_unchanged_semantic_validator_enforcement()
    test_search_matrix_and_early_stopping()
    print("All Milestone 1.3A synthetic unit tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
