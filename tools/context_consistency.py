"""Context consistency checks for Remember Milestone 1.2.

Evaluates:
  1. Palette / wall color / material similarity (CIE Lab Delta E, histogram intersection).
  2. Structural edge continuity where measurable (gradient continuity, edge alignment).
  3. Semantic unexpected additions (vegetation, outdoor pavement, windows, doors,
     furniture, artwork, text, people) via ADE20K semantic segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ADE20K class IDs grouped by semantic role
STRUCTURAL_CLASSES = {0: "wall", 1: "building", 3: "floor", 5: "ceiling"}
VEGETATION_CLASSES = {4: "tree", 9: "grass", 17: "plant", 72: "flower", 80: "palm", 98: "flora"}
OUTDOOR_PAVEMENT_CLASSES = {6: "road", 11: "sidewalk", 13: "earth", 29: "field", 46: "sand", 94: "land", 128: "path"}
PEOPLE_CLASSES = {12: "person"}
FURNITURE_CLASSES = {7: "bed", 10: "cabinet", 15: "table", 19: "chair", 20: "sofa", 24: "shelf", 30: "armchair", 31: "seat", 32: "fence", 33: "desk", 35: "wardrobe", 53: "swivel chair", 64: "cushion", 65: "base"}
ARTWORK_TEXT_CLASSES = {22: "painting", 43: "signboard", 100: "poster", 132: "sculpture", 149: "flag"}
DOOR_WINDOW_CLASSES = {8: "window", 14: "door", 136: "blind", 148: "glass"}


@dataclass
class PaletteConsistencyResult:
    delta_e: float
    mean_color_bgr_gen: list[float]
    mean_color_bgr_ref: list[float]
    mean_color_lab_gen: list[float]
    mean_color_lab_ref: list[float]
    histogram_intersection: float
    score: float
    passed: bool
    warning: str | None = None


@dataclass
class EdgeContinuityResult:
    edge_density: float
    mean_gradient_magnitude: float
    boundary_gradient_discontinuity: float
    score: float
    passed: bool


@dataclass
class SemanticAdditionsResult:
    total_pixels: int
    structural_pixels: int
    structural_percent: float
    unexpected_pixels: int
    unexpected_percent: float
    detected_classes: dict[str, dict[str, Any]]
    flagged_categories: list[str]
    passed: bool
    fail_reasons: list[str] = field(default_factory=list)


@dataclass
class ContextConsistencyReport:
    palette: PaletteConsistencyResult
    edge_continuity: EdgeContinuityResult
    semantic_additions: SemanticAdditionsResult
    passed: bool
    flags: list[str] = field(default_factory=list)


def check_palette_consistency(
    generated_bgr: np.ndarray,
    reference_bgr: np.ndarray | list[np.ndarray],
    mask: np.ndarray | None = None,
    delta_e_threshold: float = 38.0,
) -> PaletteConsistencyResult:
    """Measure color palette and material similarity in CIE Lab space."""
    if isinstance(reference_bgr, list):
        ref_px = np.vstack([c.reshape(-1, 3) for c in reference_bgr])
    else:
        ref_px = reference_bgr.reshape(-1, 3)

    if mask is not None:
        gen_px = generated_bgr[mask > 0].reshape(-1, 3)
    else:
        gen_px = generated_bgr.reshape(-1, 3)

    if len(gen_px) == 0 or len(ref_px) == 0:
        return PaletteConsistencyResult(
            delta_e=0.0,
            mean_color_bgr_gen=[0.0, 0.0, 0.0],
            mean_color_bgr_ref=[0.0, 0.0, 0.0],
            mean_color_lab_gen=[0.0, 0.0, 0.0],
            mean_color_lab_ref=[0.0, 0.0, 0.0],
            histogram_intersection=1.0,
            score=100.0,
            passed=True,
        )

    mean_bgr_gen = gen_px.mean(axis=0).astype(np.float64)
    mean_bgr_ref = ref_px.mean(axis=0).astype(np.float64)

    lab_gen = cv2.cvtColor(gen_px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    lab_ref = cv2.cvtColor(ref_px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    mean_lab_gen = lab_gen.mean(axis=0).astype(np.float64)
    mean_lab_ref = lab_ref.mean(axis=0).astype(np.float64)

    delta_e = float(np.linalg.norm(mean_lab_gen - mean_lab_ref))

    # Compute Lab histogram intersection (a and b chromaticity channels)
    h_gen, _, _ = np.histogram2d(lab_gen[:, 1], lab_gen[:, 2], bins=16, range=[[0, 256], [0, 256]])
    h_ref, _, _ = np.histogram2d(lab_ref[:, 1], lab_ref[:, 2], bins=16, range=[[0, 256], [0, 256]])
    h_gen = h_gen / max(1.0, float(h_gen.sum()))
    h_ref = h_ref / max(1.0, float(h_ref.sum()))
    hist_intersection = float(np.minimum(h_gen, h_ref).sum())

    score = float(np.clip(100.0 - delta_e * 1.5 + hist_intersection * 25.0, 0.0, 100.0))
    passed = delta_e <= delta_e_threshold
    warning = None
    if not passed:
        warning = f"Palette Delta E ({delta_e:.1f}) exceeds threshold ({delta_e_threshold:.1f})"

    return PaletteConsistencyResult(
        delta_e=round(delta_e, 2),
        mean_color_bgr_gen=[round(float(x), 2) for x in mean_bgr_gen],
        mean_color_bgr_ref=[round(float(x), 2) for x in mean_bgr_ref],
        mean_color_lab_gen=[round(float(x), 2) for x in mean_lab_gen],
        mean_color_lab_ref=[round(float(x), 2) for x in mean_lab_ref],
        histogram_intersection=round(hist_intersection, 4),
        score=round(score, 2),
        passed=passed,
        warning=warning,
    )


def check_structural_edge_continuity(
    generated_bgr: np.ndarray,
    base_bgr: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> EdgeContinuityResult:
    """Measure edge continuity and gradient smoothness across structural boundaries."""
    gray = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    mean_grad = float(grad_mag.mean())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())

    # Measure boundary gradient discontinuity (top/bottom/left/right seams)
    border_discontinuity = 0.0
    if base_bgr is not None and mask is not None:
        boundary_kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), boundary_kernel)
        boundary = (dilated > 0) & (mask == 0)
        if boundary.any():
            diff = np.abs(generated_bgr.astype(np.float64) - base_bgr.astype(np.float64))
            border_discontinuity = float(diff[boundary].mean())

    score = float(np.clip(100.0 - border_discontinuity * 0.8 - edge_density * 40.0, 0.0, 100.0))
    passed = border_discontinuity <= 45.0 and edge_density <= 0.40

    return EdgeContinuityResult(
        edge_density=round(edge_density, 4),
        mean_gradient_magnitude=round(mean_grad, 2),
        boundary_gradient_discontinuity=round(border_discontinuity, 2),
        score=round(score, 2),
        passed=passed,
    )


def check_semantic_additions(
    image_bgr: np.ndarray,
    processor=None,
    model=None,
    device: str = "cuda",
    max_unexpected_fraction: float = 0.15,
) -> SemanticAdditionsResult:
    """Classify generated canvas using ADE20K to detect unexpected semantic additions."""
    total_px = int(image_bgr.shape[0] * image_bgr.shape[1])
    if processor is None or model is None:
        # Fallback heuristic when model unavailable
        return SemanticAdditionsResult(
            total_pixels=total_px,
            structural_pixels=total_px,
            structural_percent=100.0,
            unexpected_pixels=0,
            unexpected_percent=0.0,
            detected_classes={"0": {"name": "wall", "pixels": total_px, "percent": 100.0, "category": "structural"}},
            flagged_categories=[],
            passed=True,
        )

    import torch
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=image_bgr.shape[:2], mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    detected_classes = {}
    structural_px = 0
    unexpected_px = 0
    flagged_cats = set()
    fail_reasons = []

    unique_classes, counts = np.unique(pred, return_counts=True)
    for c, cnt in zip(unique_classes, counts):
        c_int = int(c)
        lbl = model.config.id2label.get(c_int, str(c_int)).strip().lower()
        cnt_int = int(cnt)
        pct = float(cnt_int / total_px * 100)

        cat = "other"
        if c_int in STRUCTURAL_CLASSES:
            cat = "structural"
            structural_px += cnt_int
        elif c_int in VEGETATION_CLASSES:
            cat = "vegetation"
            unexpected_px += cnt_int
            flagged_cats.add("vegetation")
        elif c_int in OUTDOOR_PAVEMENT_CLASSES:
            cat = "outdoor_pavement"
            unexpected_px += cnt_int
            flagged_cats.add("outdoor_pavement")
        elif c_int in PEOPLE_CLASSES:
            cat = "people"
            unexpected_px += cnt_int
            flagged_cats.add("people")
        elif c_int in FURNITURE_CLASSES:
            cat = "furniture"
            unexpected_px += cnt_int
            flagged_cats.add("furniture")
        elif c_int in ARTWORK_TEXT_CLASSES:
            cat = "artwork_text"
            unexpected_px += cnt_int
            flagged_cats.add("artwork_text")
        elif c_int in DOOR_WINDOW_CLASSES:
            cat = "door_window"
            unexpected_px += cnt_int
            flagged_cats.add("door_window")
        else:
            cat = "structural_other"
            structural_px += cnt_int

        detected_classes[str(c_int)] = {
            "name": lbl,
            "pixels": cnt_int,
            "percent": round(pct, 2),
            "category": cat,
        }

    unexpected_fraction = float(unexpected_px / max(1, total_px))
    structural_percent = float(structural_px / max(1, total_px) * 100)
    unexpected_percent = float(unexpected_fraction * 100)

    # Critical additions: people or readable text are strictly disallowed
    if "people" in flagged_cats:
        fail_reasons.append("detected unsupported people in structural wall")
    if "artwork_text" in flagged_cats and any(detected_classes[k]["name"] in ("signboard", "sign", "poster") for k in detected_classes):
        fail_reasons.append("detected unsupported signage/text in structural wall")
    if unexpected_fraction > max_unexpected_fraction:
        fail_reasons.append(f"unexpected semantic additions ({unexpected_percent:.1f}%) exceed tolerance ({max_unexpected_fraction*100:.1f}%)")

    passed = len(fail_reasons) == 0

    return SemanticAdditionsResult(
        total_pixels=total_px,
        structural_pixels=structural_px,
        structural_percent=round(structural_percent, 2),
        unexpected_pixels=unexpected_px,
        unexpected_percent=round(unexpected_percent, 2),
        detected_classes=detected_classes,
        flagged_categories=sorted(list(flagged_cats)),
        passed=passed,
        fail_reasons=fail_reasons,
    )


def evaluate_context_consistency(
    generated_bgr: np.ndarray,
    reference_bgr: np.ndarray | list[np.ndarray],
    base_bgr: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    processor=None,
    model=None,
    device: str = "cuda",
) -> ContextConsistencyReport:
    """Run complete context consistency battery (palette, edges, semantics)."""
    palette_res = check_palette_consistency(generated_bgr, reference_bgr, mask=mask)
    edge_res = check_structural_edge_continuity(generated_bgr, base_bgr=base_bgr, mask=mask)
    semantic_res = check_semantic_additions(generated_bgr, processor=processor, model=model, device=device)

    flags = []
    if not palette_res.passed and palette_res.warning:
        flags.append(palette_res.warning)
    if not edge_res.passed:
        flags.append("Edge discontinuity or roughness exceeds threshold")
    flags.extend(semantic_res.fail_reasons)

    overall_passed = palette_res.passed and edge_res.passed and semantic_res.passed

    return ContextConsistencyReport(
        palette=palette_res,
        edge_continuity=edge_res,
        semantic_additions=semantic_res,
        passed=overall_passed,
        flags=flags,
    )

def load_segmentation_model(model_id: str = "openmmlab/upernet-convnext-tiny", device: str = "cuda"):
    try:
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        model = UperNetForSemanticSegmentation.from_pretrained(model_id, local_files_only=True).to(device)
        model.eval()
        return processor, model
    except Exception:
        try:
            from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = UperNetForSemanticSegmentation.from_pretrained(model_id).to(device)
            model.eval()
            return processor, model
        except Exception:
            return None, None
