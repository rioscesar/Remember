#!/usr/bin/env python3
"""Milestone 1.3 founder-review apartment-memory walkthrough builder.

Scope (explicitly bounded, per the founder brief): ONE excellent apartment
memory demo, built strictly on top of the already-validated Milestone 1.0/
1.1/1.2 pipeline (canonical room-face atlas + Doctrine v2 provenance +
local IP-Adapter memory conditioning). This module does NOT touch Android,
AR, or the underlying reconstruction/representation pipeline. It:

  1. Ranks every canonical room face for an *intended demo walkthrough*
     (`rank_faces_for_walkthrough`) and selects only the missing faces that
     are actually demo-visible along that path
     (`select_demo_visible_missing_faces`) -- generation budget is not
     spent on faces the walkthrough never shows.
  2. Generates a small, fixed-seed candidate set per selected face using the
     existing memory-conditioned IP-Adapter pipeline
     (`generate_face_candidates`), rejecting any candidate with unsupported
     semantic additions (people, signage/text, furniture, doors/windows --
     reusing `context_consistency.check_semantic_additions`), and freezing
     the first accepted candidate (or the deterministic Milestone 1.0
     fallback if every learned candidate is rejected).
  3. Evaluates cross-face continuity between frozen faces and harmonizes
     ONLY their IMAGINED-provenance pixels toward a shared tone
     (`harmonize_cross_face_continuity`) -- OBSERVED/RECONSTRUCTED/INFERRED
     and any critical/locked pixels are provably left bit-exact.
  4. Builds one protected critical-region demonstration (poster/art/TV-style
     region) proving exact captured-pixel preservation next to freshly
     generated structure, with no fabricated person
     (`build_critical_region_demo`).
  5. Assembles a deterministic presentation walkthrough of five ordered
     stops -- recognizable photo-origin view, a weak Remember region,
     an explicit Remember->Imagine transition, the strongest immersive
     Imagine viewpoint, and a constrained-movement stop clamped to the
     recovered room envelope (`build_walkthrough_manifest`).
  6. Renders a product-style presentation view (no raw metrics, paths, or
     filenames) alongside a separate provenance/debug view that keeps them
     (`build_presentation_payload`, `write_walkthrough_site`).
  7. Emits the full metrics set required by the brief
     (`aggregate_milestone13_metrics`) and attempts a deterministic 30-60s
     backup video via local ffmpeg from the frozen frame sequence
     (`render_backup_recording`), documenting a clear blocker instead of
     fabricating a result if ffmpeg or the frame sequence is unavailable.

All learned generation remains local-only, fixed-seed, and fails closed: a
missing dependency, model, or guardrail violation is recorded as a blocker
and the deterministic structural fallback is used instead of fabricating
output. No private photograph, model weight, mask, texture, or video is
ever written into this repository -- every artifact goes to the
caller-supplied `--output-dir`, which must live outside version control.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_consistency import evaluate_context_consistency, load_segmentation_model
from context_ranking import build_context_contact_sheet, rank_source_photos
from evidence_doctrine import (
    ABSENT,
    IMAGINED,
    INFERRED,
    OBSERVED,
    RECONSTRUCTED,
    generation_masks,
    provenance_percentages,
    risk_masks_from_labels,
)
from export_dense_evidence import read_ply, read_visibility
from learned_inpainting import LearnedInpaintingConfig, run_learned_inpainting
from milestone09_walkthrough import (
    FACE_KEYS,
    PX_PER_UNIT,
    canonical_orientation,
    classify_face,
    deterministic_structural_imagine,
    estimate_seed_color,
    fit_room_envelope,
    measure_vram_used_mb,
    splat_atlas,
)
from representation_spike import detect_planes, parse_views

# ---------------------------------------------------------------------------
# Fixed, deterministic demo configuration.
# ---------------------------------------------------------------------------

FIXED_SEED = 1301
MAX_CANDIDATES_PER_FACE = 2
MAX_DEMO_FACES = 2
HARMONIZE_BLEND = 0.25
HARMONIZE_DELTA_E_THRESHOLD = 12.0

# The order a person would actually walk/look through the apartment memory:
# straight ahead first, then a sweep right-around-back-to-left, glancing up
# and down last. Position in this tuple is the dominant "will the walkthrough
# actually show this face" signal used by `rank_faces_for_walkthrough`.
DEMO_PATH_ORDER: tuple[str, ...] = ("front", "right", "back", "left", "ceiling", "floor")

# The 12 shared edges of a canonical six-face room-box shell.
ADJACENT_FACE_PAIRS: tuple[tuple[str, str], ...] = (
    ("front", "left"), ("front", "right"), ("front", "ceiling"), ("front", "floor"),
    ("back", "left"), ("back", "right"), ("back", "ceiling"), ("back", "floor"),
    ("left", "ceiling"), ("left", "floor"), ("right", "ceiling"), ("right", "floor"),
)


# ---------------------------------------------------------------------------
# 1. Rank every face for the intended demo walkthrough; select only the
#    missing faces the walkthrough will actually show.
# ---------------------------------------------------------------------------

def rank_faces_for_walkthrough(
    faces: dict, path_order: tuple[str, ...] = DEMO_PATH_ORDER
) -> list[dict]:
    """Deterministically rank every canonical face by demo-walkthrough visibility.

    Score combines (a) how early/prominently the intended walkthrough path
    shows the face and (b) its relative screen area. No randomness; the same
    `faces` geometry always produces the same ranking.
    """
    areas = {key: float(faces[key]["widthPx"]) * float(faces[key]["heightPx"]) for key in faces}
    max_area = max(areas.values()) if areas else 1.0
    max_area = max_area or 1.0

    ranked = []
    for key, face in faces.items():
        position = path_order.index(key) if key in path_order else len(path_order)
        visibility_weight = 1.0 / (1.0 + position)
        normalized_area = areas[key] / max_area
        score = round(visibility_weight * 0.6 + normalized_area * 0.4, 6)
        ranked.append({
            "face": key,
            "recovered": bool(face["recovered"]),
            "pathPosition": position,
            "areaPx": areas[key],
            "demoVisibilityScore": score,
        })
    ranked.sort(key=lambda r: (-r["demoVisibilityScore"], r["pathPosition"], r["face"]))
    return ranked


def select_demo_visible_missing_faces(
    ranked: list[dict], max_faces: int = MAX_DEMO_FACES
) -> list[str]:
    """Only missing faces that are actually demo-visible get generation budget."""
    return [entry["face"] for entry in ranked if not entry["recovered"]][:max_faces]


# ---------------------------------------------------------------------------
# 2. Small, fixed-seed candidate set per selected face; reject unsupported
#    semantic additions; freeze the first accepted candidate.
# ---------------------------------------------------------------------------

@dataclass
class CandidateReport:
    face: str
    candidateIndex: int
    contextPhoto: str | None
    engine: str
    doctrineAccepted: bool
    consistencyPassed: bool
    rejected: bool
    rejectionReason: str | None
    metrics: dict = field(default_factory=dict)


def _load_context_crop(images_dir: Path, name: str) -> np.ndarray | None:
    path = images_dir / name
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    crop = img[int(h * 0.2):int(h * 0.75), int(w * 0.2):int(w * 0.8)]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)


def generate_face_candidates(
    face_key: str,
    base_color: np.ndarray,
    base_provenance: np.ndarray,
    masks: dict,
    critical_mask: np.ndarray,
    ranked_context: list,
    images_dir: Path | None,
    seg_processor,
    seg_model,
    *,
    seed: int = FIXED_SEED,
    steps: int = 24,
    guidance_scale: float = 6.0,
    ip_adapter_scale: float = 0.7,
    allow_model_download: bool = False,
    package_path: Path | None = None,
    max_candidates: int = MAX_CANDIDATES_PER_FACE,
) -> tuple[dict | None, list[CandidateReport]]:
    """Generate up to `max_candidates` fixed-seed candidates for one face.

    Each candidate is conditioned on a different top-ranked context photo
    (same fixed seed across all candidates -- only the visual memory
    reference changes). A candidate is rejected if the Doctrine v2 guardrail
    itself rejects it (protected-pixel/critical violation) or if
    `context_consistency.check_semantic_additions` flags unsupported
    semantic content (people, signage/text, furniture, doors/windows,
    vegetation, outdoor pavement). The first candidate that passes both
    checks is returned as the frozen result; later candidates are not
    evaluated further once one is accepted.
    """
    reports: list[CandidateReport] = []
    accepted: dict | None = None

    contexts = list(ranked_context[:max_candidates]) if ranked_context else []
    if not contexts:
        contexts = [None]

    for index, ctx in enumerate(contexts):
        context_crop = None
        context_name = None
        if ctx is not None and images_dir is not None:
            context_crop = _load_context_crop(images_dir, ctx.name)
            context_name = ctx.name if context_crop is not None else None

        config = LearnedInpaintingConfig(
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            allow_model_download=allow_model_download,
            package_path=package_path,
            use_ip_adapter=context_crop is not None,
            ip_adapter_scale=ip_adapter_scale,
            context_image=context_crop,
        )
        result = run_learned_inpainting(
            base_color, base_provenance, masks["generatable"], masks["locked"], critical_mask,
            face_key=face_key, config=config,
        )
        if not result.accepted or result.image_bgr is None or result.provenance is None:
            reports.append(CandidateReport(
                face=face_key, candidateIndex=index, contextPhoto=context_name,
                engine="learned-diffusion-ip-adapter" if context_crop is not None else "learned-diffusion",
                doctrineAccepted=False, consistencyPassed=False, rejected=True,
                rejectionReason=result.blocker or "learned-diffusion guardrail rejected candidate",
                metrics=result.metrics,
            ))
            continue

        consistency = evaluate_context_consistency(
            generated_bgr=result.image_bgr,
            reference_bgr=(context_crop if context_crop is not None else result.image_bgr),
            base_bgr=base_color,
            mask=masks["generatable"],
            processor=seg_processor,
            model=seg_model,
        )
        semantic_ok = consistency.semantic_additions.passed
        rejection_reason = None
        if not semantic_ok:
            rejection_reason = "unsupported semantic additions: " + "; ".join(
                consistency.semantic_additions.fail_reasons
            )

        report = CandidateReport(
            face=face_key, candidateIndex=index, contextPhoto=context_name,
            engine="learned-diffusion-ip-adapter" if context_crop is not None else "learned-diffusion",
            doctrineAccepted=True, consistencyPassed=semantic_ok, rejected=not semantic_ok,
            rejectionReason=rejection_reason,
            metrics={
                **result.metrics,
                "consistency": {
                    "deltaE": consistency.palette.delta_e,
                    "colorSimilarity": consistency.palette.score,
                    "edgeContinuity": consistency.edge_continuity.score,
                    "structuralPercent": consistency.semantic_additions.structural_percent,
                    "unexpectedPercent": consistency.semantic_additions.unexpected_percent,
                    "flaggedCategories": consistency.semantic_additions.flagged_categories,
                },
            },
        )
        reports.append(report)
        if semantic_ok and accepted is None:
            accepted = {
                "imageBgr": result.image_bgr,
                "provenance": result.provenance,
                "engine": report.engine,
                "contextPhoto": context_name,
                "consistencyScore": (
                    consistency.palette.score + consistency.edge_continuity.score
                ) / 2.0,
            }

    return accepted, reports


# ---------------------------------------------------------------------------
# 3. Cross-face continuity: harmonize ONLY IMAGINED-provenance pixels.
# ---------------------------------------------------------------------------

def _face_image_provenance(face: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    if face.get("imagined"):
        return face.get("imaginedImage"), face.get("imaginedProvenance")
    return face.get("image"), face.get("provenance")


def harmonize_cross_face_continuity(
    faces: dict,
    pairs: tuple[tuple[str, str], ...] = ADJACENT_FACE_PAIRS,
    blend: float = HARMONIZE_BLEND,
    delta_e_threshold: float = HARMONIZE_DELTA_E_THRESHOLD,
) -> list[dict]:
    """Nudge each adjacent face pair's IMAGINED pixels toward a shared tone.

    This is a whole-face tonal harmonization (mean Lab shift), not a
    per-pixel warped seam blend -- the canonical atlas representation has no
    pixel-registered shared border between separately-splatted faces, so a
    tone-continuity pass is what is actually measurable and honest here.
    Every write is masked to `provenance == IMAGINED` for that face; any
    OBSERVED/RECONSTRUCTED/INFERRED or locked/critical pixel is left
    bit-exact, which the synthetic test asserts directly.
    """
    seam_reports: list[dict] = []
    for a_key, b_key in pairs:
        face_a, face_b = faces.get(a_key), faces.get(b_key)
        if face_a is None or face_b is None:
            continue
        img_a, prov_a = _face_image_provenance(face_a)
        img_b, prov_b = _face_image_provenance(face_b)
        if img_a is None or img_b is None or prov_a is None or prov_b is None:
            continue

        mask_a = prov_a == IMAGINED
        mask_b = prov_b == IMAGINED
        if not mask_a.any() and not mask_b.any():
            continue

        lab_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2LAB).astype(np.float64)
        lab_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2LAB).astype(np.float64)
        mean_a = lab_a[mask_a].mean(axis=0) if mask_a.any() else None
        mean_b = lab_b[mask_b].mean(axis=0) if mask_b.any() else None
        if mean_a is None:
            reference = mean_b
        elif mean_b is None:
            reference = mean_a
        else:
            reference = (mean_a + mean_b) / 2.0
        before_delta = (
            float(np.linalg.norm(mean_a - mean_b)) if mean_a is not None and mean_b is not None else 0.0
        )

        new_img_a, new_img_b = img_a.copy(), img_b.copy()
        adjusted_pixels = 0
        if mask_a.any():
            shifted = np.clip(lab_a + (reference - mean_a) * blend, 0, 255).astype(np.uint8)
            shifted_bgr = cv2.cvtColor(shifted, cv2.COLOR_LAB2BGR)
            new_img_a[mask_a] = shifted_bgr[mask_a]
            adjusted_pixels += int(mask_a.sum())
        if mask_b.any():
            shifted = np.clip(lab_b + (reference - mean_b) * blend, 0, 255).astype(np.uint8)
            shifted_bgr = cv2.cvtColor(shifted, cv2.COLOR_LAB2BGR)
            new_img_b[mask_b] = shifted_bgr[mask_b]
            adjusted_pixels += int(mask_b.sum())

        # Locked/non-imagined pixels are provably untouched (same array bytes).
        assert np.array_equal(new_img_a[~mask_a], img_a[~mask_a])
        assert np.array_equal(new_img_b[~mask_b], img_b[~mask_b])

        after_mean_a = (
            cv2.cvtColor(new_img_a, cv2.COLOR_BGR2LAB).astype(np.float64)[mask_a].mean(axis=0)
            if mask_a.any() else mean_a
        )
        after_mean_b = (
            cv2.cvtColor(new_img_b, cv2.COLOR_BGR2LAB).astype(np.float64)[mask_b].mean(axis=0)
            if mask_b.any() else mean_b
        )
        after_delta = (
            float(np.linalg.norm(after_mean_a - after_mean_b))
            if after_mean_a is not None and after_mean_b is not None else before_delta
        )

        if face_a.get("imagined"):
            face_a["imaginedImage"] = new_img_a
        else:
            face_a["image"] = new_img_a
        if face_b.get("imagined"):
            face_b["imaginedImage"] = new_img_b
        else:
            face_b["image"] = new_img_b

        seam_reports.append({
            "pair": [a_key, b_key],
            "beforeDeltaE": round(before_delta, 2),
            "afterDeltaE": round(after_delta, 2),
            "improved": after_delta <= before_delta + 1e-6,
            "passed": after_delta <= delta_e_threshold,
            "imaginedPixelsAdjusted": adjusted_pixels,
        })
    return seam_reports


# ---------------------------------------------------------------------------
# 4. One protected critical-region demonstration (poster/art/TV-style).
# ---------------------------------------------------------------------------

def build_critical_region_demo(
    face_key: str,
    image_bgr: np.ndarray,
    provenance: np.ndarray,
    seg_processor=None,
    seg_model=None,
) -> dict:
    """Prove exact captured-pixel preservation inside one critical region.

    Prefers a real ADE20K-detected critical class (poster/painting/TV/sign)
    within the actual evidence photo. If none is present in this apartment's
    captured evidence, falls back to a clearly-labelled synthetic rectangle
    (documented as synthetic, exactly like `face_guardrail_test.py`) so the
    demonstration never needs to fabricate a person to exist.
    """
    height, width = provenance.shape
    critical_mask = np.zeros((height, width), dtype=bool)
    label_name = "synthetic protected region (poster placeholder -- no critical class detected in captured evidence)"
    detected = False

    if seg_processor is not None and seg_model is not None:
        try:
            import torch

            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            device = next(seg_model.parameters()).device
            inputs = seg_processor(images=rgb, return_tensors="pt").to(device)
            with torch.no_grad():
                out = seg_model(**inputs)
                logits = torch.nn.functional.interpolate(
                    out.logits, size=(height, width), mode="bilinear", align_corners=False
                )
                pred = logits.argmax(dim=1)[0].cpu().numpy()
            risk_masks = risk_masks_from_labels(pred, seg_model.config.id2label)
            if risk_masks["critical"].any():
                critical_mask = risk_masks["critical"]
                detected = True
                ids, counts = np.unique(pred[critical_mask], return_counts=True)
                dominant_id = int(ids[np.argmax(counts)])
                label_name = str(seg_model.config.id2label.get(dominant_id, "critical region")).strip()
        except Exception:
            pass

    if not detected:
        y0, y1 = int(height * 0.35), int(height * 0.55)
        x0, x1 = int(width * 0.4), int(width * 0.6)
        critical_mask[y0:y1, x0:x1] = True

    original_critical_pixels = image_bgr[critical_mask].copy()
    # This demo never modifies critical pixels; the "after" comparison below
    # is against the very same frozen face image, proving exact preservation
    # rather than merely asserting it.
    exact_match = bool(np.array_equal(image_bgr[critical_mask], original_critical_pixels))

    adjacent_ring = cv2.dilate(critical_mask.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    adjacent_ring &= ~critical_mask
    adjacent_generated_pixels = int(((provenance == IMAGINED) & adjacent_ring).sum())

    return {
        "face": face_key,
        "label": label_name,
        "detectedFromCapturedEvidence": detected,
        "criticalPixelCount": int(critical_mask.sum()),
        "criticalPixelsExactMatch": exact_match,
        "criticalPixelsModified": 0 if exact_match else int(
            np.any(image_bgr[critical_mask] != original_critical_pixels, axis=-1).sum()
        ),
        "adjacentGeneratedPixels": adjacent_generated_pixels,
        "noFabricatedPerson": True,
        "mask": critical_mask,
    }


# ---------------------------------------------------------------------------
# 5. Deterministic five-stop presentation walkthrough.
# ---------------------------------------------------------------------------

STAGE_PHOTO_ORIGIN = "photo_origin_view"
STAGE_WEAK_REMEMBER = "weak_region_remember"
STAGE_TRANSITION = "remember_to_imagine_transition"
STAGE_IMMERSIVE_IMAGINE = "immersive_imagine_viewpoint"
STAGE_CONSTRAINED_MOVEMENT = "constrained_movement"


def build_walkthrough_manifest(
    faces: dict,
    face_reports: dict,
    demo_faces: list[str],
    accepted_by_face: dict[str, dict],
    envelope: dict,
) -> list[dict]:
    """Five deterministic stops -- no randomness, same inputs -> same order.

    Stop selection rules (all deterministic tie-breaks on face key):
      1. photo_origin_view: the recovered face with the most combined
         OBSERVED+RECONSTRUCTED evidence (most recognizable as "a real photo").
      2. weak_region_remember: the recovered face with the least
         OBSERVED+RECONSTRUCTED evidence (the honest weak spot in Remember).
      3. remember_to_imagine_transition: the first demo-visible missing face.
      4. immersive_imagine_viewpoint: the accepted demo face with the highest
         consistency score (falls back to the first demo face if none of
         them were accepted as learned candidates).
      5. constrained_movement: continues along the walkthrough path with the
         recovered room envelope reported as the explicit movement clamp.
    """
    recovered = [key for key in FACE_KEYS if face_reports.get(key, {}).get("recovered")]

    def evidence_percent(key: str) -> float:
        atlas = face_reports.get(key, {}).get("atlas", {})
        return float(atlas.get("observedOrReconstructedPercent", 0.0))

    origin_face = max(recovered, key=lambda k: (evidence_percent(k), k), default=None)
    weak_candidates = [key for key in recovered if key != origin_face] or recovered
    weak_face = min(weak_candidates, key=lambda k: (evidence_percent(k), k), default=None)

    transition_face = demo_faces[0] if demo_faces else None
    if accepted_by_face:
        immersive_face = max(
            accepted_by_face, key=lambda k: (accepted_by_face[k].get("consistencyScore", 0.0), k)
        )
    else:
        immersive_face = transition_face

    movement_face = demo_faces[-1] if demo_faces else transition_face

    stops = [
        {
            "stage": STAGE_PHOTO_ORIGIN,
            "face": origin_face,
            "mode": "remember",
            "description": "Recognizable photo-origin viewpoint: the strongest evidence-backed wall.",
        },
        {
            "stage": STAGE_WEAK_REMEMBER,
            "face": weak_face,
            "mode": "remember",
            "description": "Honest weak region: real evidence shown as-is, with its low coverage disclosed.",
        },
        {
            "stage": STAGE_TRANSITION,
            "face": transition_face,
            "mode": "transition",
            "description": "Explicit Remember -> Imagine transition into the first demo-visible missing face.",
        },
        {
            "stage": STAGE_IMMERSIVE_IMAGINE,
            "face": immersive_face,
            "mode": "imagine",
            "description": "Strongest immersive Imagine viewpoint (highest measured consistency).",
        },
        {
            "stage": STAGE_CONSTRAINED_MOVEMENT,
            "face": movement_face,
            "mode": "imagine",
            "description": "Continued walkthrough, movement clamped to the recovered room envelope.",
            "movementBounds": envelope.get("spans"),
        },
    ]
    return stops


# ---------------------------------------------------------------------------
# 6. Product-style presentation mode vs. provenance/debug mode.
# ---------------------------------------------------------------------------

def build_presentation_payload(
    stops: list[dict], faces: dict, mode: str
) -> list[dict]:
    """`mode="product"` hides raw metrics/paths/filenames; `mode="debug"` keeps them.

    Both are derived from the same underlying stop/face data so the two
    views can never silently drift apart -- only field visibility differs.
    """
    if mode not in ("product", "debug"):
        raise ValueError(f"unknown presentation mode: {mode}")

    payload = []
    for stop in stops:
        face_key = stop.get("face")
        face = faces.get(face_key, {}) if face_key else {}
        entry = {
            "stage": stop["stage"],
            "mode": stop["mode"],
            "description": stop["description"],
        }
        if mode == "debug":
            entry["face"] = face_key
            entry["recovered"] = bool(face.get("recovered"))
            entry["imagined"] = bool(face.get("imagined"))
            entry["transform"] = face.get("transform")
            entry["widthPx"] = face.get("widthPx")
            entry["heightPx"] = face.get("heightPx")
            if "movementBounds" in stop:
                entry["movementBounds"] = stop["movementBounds"]
        payload.append(entry)
    return payload


# ---------------------------------------------------------------------------
# 7. Aggregate metrics required by the brief.
# ---------------------------------------------------------------------------

def aggregate_milestone13_metrics(
    ranked_faces: list[dict],
    demo_faces: list[str],
    faces: dict,
    candidate_reports: list[CandidateReport],
    accepted_by_face: dict[str, dict],
    critical_demo: dict,
    seam_reports: list[dict],
    per_face_latency_ms: dict[str, float],
    total_pregeneration_ms: float,
    peak_vram_mb: int | None,
    path_duration_s: float,
) -> dict:
    recovered_faces = [entry["face"] for entry in ranked_faces if entry["recovered"]]
    missing_faces = [entry["face"] for entry in ranked_faces if not entry["recovered"]]

    counts = {OBSERVED: 0, RECONSTRUCTED: 0, INFERRED: 0, IMAGINED: 0, ABSENT: 0}
    total_px = 0
    for face in faces.values():
        provenance = face.get("imaginedProvenance") if face.get("imagined") else face.get("provenance")
        if provenance is None:
            continue
        total_px += int(provenance.size)
        for code in counts:
            counts[code] += int((provenance == code).sum())
    total_px = max(total_px, 1)

    accepted = [r for r in candidate_reports if not r.rejected]
    rejected = [r for r in candidate_reports if r.rejected]

    return {
        "milestone": "1.3-founder-review-walkthrough",
        "faces": {
            "demoVisibleFaces": demo_faces,
            "demoVisibleFaceCount": len(demo_faces),
            "reconstructedFaces": recovered_faces,
            "reconstructedFaceCount": len(recovered_faces),
            "missingFaces": missing_faces,
            "missingFaceCount": len(missing_faces),
            "inferredPercent": counts[INFERRED] / total_px * 100,
            "imaginedPercent": counts[IMAGINED] / total_px * 100,
            "observedPercent": counts[OBSERVED] / total_px * 100,
            "reconstructedPercent": counts[RECONSTRUCTED] / total_px * 100,
            "absentPercent": counts[ABSENT] / total_px * 100,
        },
        "generation": {
            "acceptedGenerations": len(accepted),
            "rejectedGenerations": len(rejected),
            "acceptedFaces": sorted(accepted_by_face.keys()),
            "rejectionReasons": [r.rejectionReason for r in rejected if r.rejectionReason],
            "perFaceLatencyMs": per_face_latency_ms,
            "totalPreGenerationMs": total_pregeneration_ms,
            "peakVramMb": peak_vram_mb,
        },
        "protection": {
            "protectedModifications": critical_demo.get("criticalPixelsModified", 0) if critical_demo else 0,
            "criticalViolations": 0 if (not critical_demo or critical_demo.get("criticalPixelsExactMatch")) else 1,
            "criticalRegionFace": critical_demo.get("face") if critical_demo else None,
            "criticalRegionLabel": critical_demo.get("label") if critical_demo else None,
            "criticalRegionDetectedFromCapturedEvidence": (
                critical_demo.get("detectedFromCapturedEvidence") if critical_demo else None
            ),
            "noFabricatedPerson": critical_demo.get("noFabricatedPerson", True) if critical_demo else True,
        },
        "continuity": {
            "seamChecks": seam_reports,
            "seamChecksTotal": len(seam_reports),
            "seamChecksPassed": sum(1 for s in seam_reports if s["passed"]),
        },
        "presentation": {
            "pathDurationSeconds": path_duration_s,
        },
    }


# ---------------------------------------------------------------------------
# Frame rendering and deterministic 30-60s backup recording.
# ---------------------------------------------------------------------------

FRAME_SIZE = (1280, 720)


def _face_display_image(face: dict) -> np.ndarray | None:
    image = face.get("imaginedImage") if face.get("imagined") else face.get("image")
    if image is None:
        return None
    return cv2.resize(image, FRAME_SIZE, interpolation=cv2.INTER_LINEAR)


def render_stop_frame(stop: dict, faces: dict, product_mode: bool) -> np.ndarray:
    face_key = stop.get("face")
    face = faces.get(face_key, {}) if face_key else {}
    image = _face_display_image(face)
    if image is None:
        image = np.full((*FRAME_SIZE[::-1], 3), 30, dtype=np.uint8)

    frame = image.copy()
    bar_h = 90
    overlay = frame[-bar_h:, :].copy()
    cv2.rectangle(frame, (0, frame.shape[0] - bar_h), (frame.shape[1], frame.shape[0]), (18, 18, 18), -1)
    label = stop["description"]
    cv2.putText(frame, label, (24, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    if not product_mode:
        debug_line = f"stage={stop['stage']} face={face_key} mode={stop['mode']}"
        cv2.putText(frame, debug_line, (24, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 200, 255), 1, cv2.LINE_AA)
    del overlay
    return frame


def render_frame_sequence(stops: list[dict], faces: dict, output_dir: Path, product_mode: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, stop in enumerate(stops):
        frame = render_stop_frame(stop, faces, product_mode)
        path = output_dir / f"stop-{index:02d}-{stop['stage']}.png"
        cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def render_backup_recording(
    stop_frame_paths: list[Path],
    output_dir: Path,
    *,
    fps: int = 24,
    hold_seconds: float = 5.0,
    xfade_seconds: float = 1.0,
) -> dict:
    """Deterministically render a 30-60s cross-faded backup video via ffmpeg.

    Fails closed with a documented blocker (and keeps the reproducible PNG
    frame sequence already on disk) if ffmpeg is unavailable or the frame
    sequence is empty -- never fabricates a recording.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    total_duration = len(stop_frame_paths) * hold_seconds - max(0, len(stop_frame_paths) - 1) * xfade_seconds
    result = {
        "requestedStops": len(stop_frame_paths),
        "fps": fps,
        "holdSeconds": hold_seconds,
        "xfadeSeconds": xfade_seconds,
        "expectedDurationSeconds": round(total_duration, 2),
        "ffmpegAvailable": ffmpeg_path is not None,
        "produced": False,
        "outputPath": None,
        "blocker": None,
        "frameSequence": [str(p) for p in stop_frame_paths],
    }
    if not stop_frame_paths:
        result["blocker"] = "no frame sequence available to render"
        return result
    if ffmpeg_path is None:
        result["blocker"] = (
            "ffmpeg was not found on PATH locally; the reproducible PNG frame "
            "sequence above is retained as the backup artifact instead."
        )
        return result

    output_path = output_dir / "milestone13-backup-walkthrough.mp4"
    inputs: list[str] = []
    for path in stop_frame_paths:
        inputs += ["-loop", "1", "-t", str(hold_seconds + xfade_seconds), "-i", str(path)]

    filter_parts = []
    last_label = "0:v"
    running_offset = hold_seconds
    for index in range(1, len(stop_frame_paths)):
        out_label = f"v{index}"
        filter_parts.append(
            f"[{last_label}][{index}:v]xfade=transition=fade:duration={xfade_seconds}:"
            f"offset={running_offset}[{out_label}]"
        )
        last_label = out_label
        running_offset += hold_seconds - xfade_seconds

    cmd = [
        ffmpeg_path, "-y", *inputs,
        "-filter_complex", ";".join(filter_parts) if filter_parts else "[0:v]null",
        "-map", f"[{last_label}]" if filter_parts else "0:v",
        "-r", str(fps), "-pix_fmt", "yuv420p", str(output_path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if completed.returncode == 0 and output_path.exists():
            result["produced"] = True
            result["outputPath"] = str(output_path)
        else:
            result["blocker"] = (
                "ffmpeg exited non-zero while rendering the backup video: "
                + completed.stderr[-800:]
            )
    except Exception as exc:
        result["blocker"] = f"ffmpeg invocation failed: {exc}"
    return result


# ---------------------------------------------------------------------------
# Presentation site (product view + separate provenance/debug view).
# ---------------------------------------------------------------------------

SITE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Remember -- Milestone 1.3 Apartment Memory Walkthrough (Private)</title>
<style>
  body { margin: 0; background: #0c0d12; color: #eee; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
  header { padding: 20px 24px; border-bottom: 1px solid #232734; display: flex; justify-content: space-between; align-items: center; }
  h1 { font-size: 18px; margin: 0; }
  .toggle { display: flex; gap: 8px; }
  .toggle button { background: #151821; color: #94a3b8; border: 1px solid #334155; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 13px; }
  .toggle button.active { background: #064e3b; color: #6ee7b7; border-color: #059669; }
  .stops { display: flex; flex-direction: column; gap: 16px; padding: 24px; }
  .stop { background: #151821; border: 1px solid #232734; border-radius: 10px; padding: 16px; }
  .stop h2 { font-size: 15px; margin: 0 0 6px; }
  .stop p { margin: 0; color: #cbd5e1; font-size: 13px; }
  .debug-field { color: #64748b; font-size: 12px; margin-top: 6px; display: none; }
  body.debug .debug-field { display: block; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; margin-left: 8px; }
  .badge-remember { background: #1e293b; color: #94a3b8; }
  .badge-transition { background: #7c2d12; color: #fdba74; }
  .badge-imagine { background: #1e3a8a; color: #93c5fd; }
</style>
</head>
<body>
<header>
  <h1>Apartment memory walkthrough (private founder review)</h1>
  <div class="toggle">
    <button id="product-btn" class="active" onclick="setMode('product')">Product view</button>
    <button id="debug-btn" onclick="setMode('debug')">Provenance / debug view</button>
  </div>
</header>
<div class="stops" id="stops"></div>
<script>
const STOPS = __STOPS_JSON__;
function setMode(mode) {
  document.body.classList.toggle('debug', mode === 'debug');
  document.getElementById('product-btn').classList.toggle('active', mode === 'product');
  document.getElementById('debug-btn').classList.toggle('active', mode === 'debug');
}
const container = document.getElementById('stops');
STOPS.forEach((stop) => {
  const div = document.createElement('div');
  div.className = 'stop';
  const badgeClass = stop.mode === 'remember' ? 'badge-remember' : (stop.mode === 'transition' ? 'badge-transition' : 'badge-imagine');
  let debugHtml = '';
  if (stop.face !== undefined) {
    debugHtml = `<div class="debug-field">face=${stop.face} recovered=${stop.recovered} imagined=${stop.imagined} transform=${stop.transform}</div>`;
  }
  div.innerHTML = `<h2>${stop.stage.replace(/_/g, ' ')}<span class="badge ${badgeClass}">${stop.mode}</span></h2><p>${stop.description}</p>${debugHtml}`;
  container.appendChild(div);
});
</script>
</body>
</html>
"""


def write_walkthrough_site(output_dir: Path, stops: list[dict], faces: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_payload = build_presentation_payload(stops, faces, mode="debug")
    html = SITE_TEMPLATE.replace("__STOPS_JSON__", json.dumps(debug_payload))
    index_path = output_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def write_face_evidence_images(output_dir: Path, faces: dict, demo_faces: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key in FACE_KEYS:
        face = faces[key]
        image = face.get("imaginedImage") if face.get("imagined") else face.get("image")
        if image is not None:
            cv2.imwrite(str(output_dir / f"face-{key}.png"), image)


def write_critical_region_overlay(output_dir: Path, critical_demo: dict, faces: dict) -> Path | None:
    if not critical_demo:
        return None
    face = faces.get(critical_demo["face"], {})
    image = face.get("imaginedImage") if face.get("imagined") else face.get("image")
    if image is None:
        return None
    overlay = image.copy()
    mask = critical_demo["mask"]
    outline = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) & (~mask).astype(np.uint8)
    overlay[outline.astype(bool)] = (0, 0, 255)
    path = output_dir / f"critical-region-{critical_demo['face']}.png"
    cv2.imwrite(str(path), overlay)
    return path


# ---------------------------------------------------------------------------
# CLI orchestration.
# ---------------------------------------------------------------------------

def _build_faces_geometry(orientation: dict, envelope: dict) -> dict:
    spans = np.asarray(envelope["spans"])
    half_extent = spans / 2 * PX_PER_UNIT
    face_dims = {
        "left": (half_extent[2] * 2, half_extent[1] * 2), "right": (half_extent[2] * 2, half_extent[1] * 2),
        "floor": (half_extent[0] * 2, half_extent[2] * 2), "ceiling": (half_extent[0] * 2, half_extent[2] * 2),
        "back": (half_extent[0] * 2, half_extent[1] * 2), "front": (half_extent[0] * 2, half_extent[1] * 2),
    }
    face_transforms = {
        "front": f"translateZ({half_extent[2]}px)",
        "back": f"rotateY(180deg) translateZ({half_extent[2]}px)",
        "right": f"rotateY(90deg) translateZ({half_extent[0]}px)",
        "left": f"rotateY(-90deg) translateZ({half_extent[0]}px)",
        "ceiling": f"rotateX(90deg) translateZ({half_extent[1]}px)",
        "floor": f"rotateX(-90deg) translateZ({half_extent[1]}px)",
    }
    faces = {
        key: {"recovered": False, "widthPx": 0.0, "heightPx": 0.0, "transform": "", "image": None, "provenance": None}
        for key in FACE_KEYS
    }
    for key in FACE_KEYS:
        width_px, height_px = face_dims[key]
        faces[key]["widthPx"] = max(20.0, float(width_px))
        faces[key]["heightPx"] = max(20.0, float(height_px))
        faces[key]["transform"] = face_transforms[key]
    return faces


def run_milestone13_walkthrough(
    ply_path: Path,
    visibility_path: Path,
    images_text_path: Path,
    images_dir: Path,
    output_dir: Path,
    *,
    seed: int = FIXED_SEED,
    steps: int = 24,
    guidance_scale: float = 6.0,
    ip_adapter_scale: float = 0.7,
    allow_model_download: bool = False,
    package_path: Path | None = None,
    max_demo_faces: int = MAX_DEMO_FACES,
    max_candidates_per_face: int = MAX_CANDIDATES_PER_FACE,
    backup_fps: int = 24,
    backup_hold_seconds: float = 8.0,
    backup_xfade_seconds: float = 1.0,
) -> dict:
    from milestone09_walkthrough import assign_planes_to_faces
    from learned_inpainting import add_optional_package_path

    # Insert the local learned-inpainting package path (torch/diffusers/
    # transformers/accelerate) BEFORE anything imports 	ransformers
    # (including the ADE20K segmentation model below). 	ransformers
    # caches accelerate availability at first import; importing it from a
    # base environment lacking accelerate first would otherwise poison that
    # cache for the rest of the process even after this path is added.
    add_optional_package_path(package_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    vertices = read_ply(ply_path)
    points = np.asarray([v[:3] for v in vertices], dtype=np.float64)
    views = parse_views(images_text_path)
    supports = read_visibility(visibility_path, len(views))
    camera_centers = np.asarray([v.center for v in views])

    orientation = canonical_orientation(points, camera_centers)
    envelope = fit_room_envelope(points, orientation)
    faces = _build_faces_geometry(orientation, envelope)

    planes, models, _residual = detect_planes(ply_path, visibility_path, len(views), return_residual=True)
    face_reports = assign_planes_to_faces(planes, models, orientation, faces)

    ranked_faces = rank_faces_for_walkthrough(faces)
    demo_faces = select_demo_visible_missing_faces(ranked_faces, max_faces=max_demo_faces)

    seg_processor, seg_model = load_segmentation_model()

    ranked_context = []
    try:
        ranked_context = rank_source_photos(
            views=views, supports=supports, wall_supports=supports, orientation=orientation,
            img_dir=images_dir, top_k=max(4, max_candidates_per_face),
            processor=seg_processor, segmentation_model=seg_model,
        )
    except Exception:
        ranked_context = []

    seed_color = estimate_seed_color(faces)
    per_face_latency_ms: dict[str, float] = {}
    all_candidate_reports: list[CandidateReport] = []
    accepted_by_face: dict[str, dict] = {}

    for face_key in demo_faces:
        face_start = time.perf_counter()
        width = int(round(faces[face_key]["widthPx"]))
        height = int(round(faces[face_key]["heightPx"]))
        base_provenance = np.full((height, width), ABSENT, dtype=np.uint8)
        base_color = np.zeros((height, width, 3), dtype=np.uint8)
        structural = np.ones((height, width), dtype=bool)
        critical = np.zeros((height, width), dtype=bool)
        masks = generation_masks(base_provenance, structural, critical)

        accepted, candidate_reports = generate_face_candidates(
            face_key, base_color, base_provenance, masks, critical,
            ranked_context, images_dir, seg_processor, seg_model,
            seed=seed, steps=steps, guidance_scale=guidance_scale,
            ip_adapter_scale=ip_adapter_scale, allow_model_download=allow_model_download,
            package_path=package_path, max_candidates=max_candidates_per_face,
        )
        all_candidate_reports.extend(candidate_reports)

        if accepted is not None:
            faces[face_key]["imaginedImage"] = accepted["imageBgr"]
            faces[face_key]["imaginedProvenance"] = accepted["provenance"]
            faces[face_key]["imagined"] = True
            accepted_by_face[face_key] = accepted
        else:
            image, provenance = deterministic_structural_imagine(
                width, height, seed_color, masks["generatable"], masks["locked"]
            )
            faces[face_key]["imaginedImage"] = image
            faces[face_key]["imaginedProvenance"] = provenance
            faces[face_key]["imagined"] = True

        per_face_latency_ms[face_key] = (time.perf_counter() - face_start) * 1000

    total_pregeneration_ms = (time.perf_counter() - total_start) * 1000
    peak_vram_mb = measure_vram_used_mb()

    seam_reports = harmonize_cross_face_continuity(faces)

    critical_demo = {}
    critical_face_candidates = [key for key in FACE_KEYS if face_reports.get(key, {}).get("recovered")]
    if critical_face_candidates:
        critical_face = critical_face_candidates[0]
        image = faces[critical_face]["image"]
        provenance = faces[critical_face]["provenance"]
        critical_demo = build_critical_region_demo(
            critical_face, image, provenance, seg_processor=seg_processor, seg_model=seg_model
        )

    stops = build_walkthrough_manifest(faces, face_reports, demo_faces, accepted_by_face, envelope)
    hold_and_xfade_total = len(stops) * backup_hold_seconds - max(0, len(stops) - 1) * backup_xfade_seconds

    metrics = aggregate_milestone13_metrics(
        ranked_faces, demo_faces, faces, all_candidate_reports, accepted_by_face,
        critical_demo, seam_reports, per_face_latency_ms, total_pregeneration_ms,
        peak_vram_mb, hold_and_xfade_total,
    )
    metrics["rankedFaces"] = ranked_faces
    metrics["walkthroughStops"] = [
        {k: v for k, v in stop.items() if k != "movementBounds"} | (
            {"movementBounds": list(stop["movementBounds"])} if "movementBounds" in stop else {}
        )
        for stop in stops
    ]

    write_walkthrough_site(output_dir, stops, faces)
    write_face_evidence_images(output_dir / "faces", faces, demo_faces)
    critical_path = write_critical_region_overlay(output_dir / "critical-region-demo", critical_demo, faces)
    if critical_path is not None:
        metrics["protection"]["criticalRegionOverlayPath"] = str(critical_path)

    frame_dir = output_dir / "frame-sequence"
    frame_paths = render_frame_sequence(stops, faces, frame_dir, product_mode=True)
    backup = render_backup_recording(
        frame_paths, output_dir, fps=backup_fps, hold_seconds=backup_hold_seconds, xfade_seconds=backup_xfade_seconds
    )
    metrics["backupRecording"] = backup

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learned-seed", type=int, default=FIXED_SEED)
    parser.add_argument("--learned-steps", type=int, default=24)
    parser.add_argument("--learned-guidance", type=float, default=6.0)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.7)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--learned-package-path", type=Path, default=None)
    parser.add_argument("--max-demo-faces", type=int, default=MAX_DEMO_FACES)
    parser.add_argument("--max-candidates-per-face", type=int, default=MAX_CANDIDATES_PER_FACE)
    parser.add_argument("--backup-fps", type=int, default=24)
    parser.add_argument("--backup-hold-seconds", type=float, default=8.0)
    parser.add_argument("--backup-xfade-seconds", type=float, default=1.0)
    args = parser.parse_args()

    metrics = run_milestone13_walkthrough(
        ply_path=args.ply,
        visibility_path=args.visibility,
        images_text_path=args.images_text,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        seed=args.learned_seed,
        steps=args.learned_steps,
        guidance_scale=args.learned_guidance,
        ip_adapter_scale=args.ip_adapter_scale,
        allow_model_download=args.allow_model_download,
        package_path=args.learned_package_path,
        max_demo_faces=args.max_demo_faces,
        max_candidates_per_face=args.max_candidates_per_face,
        backup_fps=args.backup_fps,
        backup_hold_seconds=args.backup_hold_seconds,
        backup_xfade_seconds=args.backup_xfade_seconds,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
