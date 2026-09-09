"""Shared Representation Doctrine v2 primitives for Milestone 0.8.

Defines the four evidence classes (OBSERVED, RECONSTRUCTED, INFERRED,
IMAGINED -- the last of which is never produced by this codebase), the
ADE20K risk-tier classification used to decide what may ever be
conservatively inferred, and the single bounded-inference fill routine
used everywhere inference is allowed. Keeping this logic in one shared,
tested module (rather than duplicated per script) is what lets
`face_guardrail_test.py` prove the *actual* production code path refuses
to complete identity-critical content, not a reimplementation of it.
"""

from __future__ import annotations

import numpy as np
import cv2

# --------------------------------------------------------------------------
# Evidence-class provenance codes (extends the CAPTURED_* codes already
# used by semantic_plane_spike.py / wall_coherence_spike.py in prior
# milestones -- those remain valid OBSERVED/RECONSTRUCTED codes here).
# --------------------------------------------------------------------------

ABSENT = 0                # unsupported, and either critical or beyond the inference radius
OBSERVED = 1               # a single captured photograph directly supports this pixel
RECONSTRUCTED = 2          # two or more captured photographs agree on this pixel
INFERRED = 3               # no direct capture; conservatively completed from nearby evidence
IMAGINED = 4               # never produced; reserved so provenance maps can always represent it as 0%

EVIDENCE_CLASS_NAMES = {
    ABSENT: "ABSENT",
    OBSERVED: "OBSERVED",
    RECONSTRUCTED: "RECONSTRUCTED",
    INFERRED: "INFERRED",
    IMAGINED: "IMAGINED",
}

# --------------------------------------------------------------------------
# ADE20K-150 risk classification.
#
# STRUCTURAL / low-risk: may be conservatively inferred when unsupported.
# OBJECT / medium-risk: photographed surfaces must dominate; only very
#   limited, non-appearance-changing shape completion is contemplated
#   (Phase 8 tests this separately) -- this module does not auto-infer
#   object interiors, only exposes the classification.
# CRITICAL / high-risk: must never be inferred over. Unsupported critical
#   regions stay ABSENT (fade), regardless of how small the gap is.
# --------------------------------------------------------------------------

STRUCTURAL_LABELS = {
    "wall", "floor", "flooring", "ceiling", "windowpane", "window", "door",
    "stairs", "stairway", "column", "pillar", "railing", "rail", "bannister",
    "banister", "base", "sidewalk", "pavement",
}

CRITICAL_LABELS = {
    "person", "painting", "picture", "poster", "animal", "signboard", "sign",
    "bulletin board", "clock", "flag", "sculpture", "mirror",
    "television receiver", "television", "tv", "screen", "crt screen",
}

# Everything else observed by the segmentation model (couch, table, chair,
# cabinet, lamp, etc.) defaults to OBJECT / medium-risk.


def classify_label_name(name: str) -> str:
    normalized = name.lower().strip()
    if normalized in STRUCTURAL_LABELS:
        return "structural"
    if normalized in CRITICAL_LABELS:
        return "critical"
    return "object"


def risk_masks_from_labels(label_map: np.ndarray, id2label: dict) -> dict:
    """Given a per-pixel ADE20K class-id map, return boolean masks for each
    risk tier plus the raw tier-name array (useful for diagnostics)."""
    tier_of_id = {int(i): classify_label_name(name) for i, name in id2label.items()}
    tier_lookup = np.zeros(max(tier_of_id.keys()) + 1, dtype=np.uint8)
    code = {"structural": 0, "object": 1, "critical": 2}
    for class_id, tier in tier_of_id.items():
        tier_lookup[class_id] = code[tier]
    safe_labels = np.clip(label_map, 0, tier_lookup.shape[0] - 1)
    tiers = tier_lookup[safe_labels]
    return {
        "structural": tiers == 0,
        "object": tiers == 1,
        "critical": tiers == 2,
    }


# --------------------------------------------------------------------------
# Bounded inference fill.
#
# This is the ONLY place this codebase ever fabricates a pixel that was not
# directly observed. It is deliberately conservative:
#   - critical_mask pixels are NEVER touched, even if unsupported -- they
#     stay ABSENT (alpha 0) regardless of how small the gap is.
#   - the fill is a classical (non-learned) PDE inpaint (Telea), which only
#     ever extrapolates from nearby real pixel colors -- there is no
#     learned prior, no generative model, and no semantic content added.
#   - fill is capped to a maximum distance from the nearest observed pixel
#     (max_distance_px). Beyond that distance, pixels remain ABSENT rather
#     than being extrapolated indefinitely -- this is what keeps inference
#     "structural continuity" rather than "confident invention" of regions
#     with no nearby evidence at all.
# --------------------------------------------------------------------------

def bounded_inference_fill(
    color_bgr: np.ndarray,
    supported_mask: np.ndarray,
    critical_mask: np.ndarray,
    max_distance_px: float,
    inpaint_radius: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservatively complete unsupported, non-critical pixels.

    Returns (filled_color_bgr, provenance_delta) where provenance_delta is
    INFERRED at every pixel this function filled and ABSENT everywhere else
    (including all critical gaps, which are always left ABSENT).
    """
    height, width = supported_mask.shape
    unsupported = ~supported_mask
    eligible = unsupported & ~critical_mask

    if not eligible.any():
        provenance_delta = np.zeros((height, width), dtype=np.uint8)
        return color_bgr.copy(), provenance_delta

    # Distance (in pixels) from every location to the nearest OBSERVED or
    # RECONSTRUCTED pixel: the distance transform of the inverse of
    # supported_mask (cv2.distanceTransform measures distance to the
    # nearest zero pixel in its input).
    distance_to_support = cv2.distanceTransform(
        (~supported_mask).astype(np.uint8), cv2.DIST_L2, 5
    )

    within_range = distance_to_support <= max_distance_px
    fill_mask = (eligible & within_range).astype(np.uint8) * 255

    if fill_mask.sum() == 0:
        provenance_delta = np.zeros((height, width), dtype=np.uint8)
        return color_bgr.copy(), provenance_delta

    # cv2.inpaint reads from *all* non-masked pixels in the source image,
    # including critical/unsupported ones outside fill_mask -- but those
    # were never masked as "to fill" so they are only ever read as source
    # texture, never overwritten. Zero them only if wholly unsupported so
    # they don't bleed uninitialized structure into the fill.
    source = color_bgr.copy()
    fully_unknown = unsupported & ~within_range
    source[fully_unknown] = 0

    filled = cv2.inpaint(source, fill_mask, inpaint_radius, cv2.INPAINT_TELEA)

    result = color_bgr.copy()
    fill_bool = fill_mask > 0
    result[fill_bool] = filled[fill_bool]

    provenance_delta = np.where(fill_bool, INFERRED, ABSENT).astype(np.uint8)
    return result, provenance_delta
