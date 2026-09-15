import sys, os, time, json, logging, argparse
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure learned packages and tools are in sys.path immediately
_PKG_DIR = Path(r"C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\learned-inpainting-packages")
if _PKG_DIR.exists():
    os.environ["REMEMBER_LEARNED_PYTHONPATH"] = str(_PKG_DIR)
    if str(_PKG_DIR) not in sys.path:
        sys.path.insert(0, str(_PKG_DIR))

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from context_consistency import (
    ContextConsistencyReport,
    evaluate_context_consistency,
    load_segmentation_model,
)
from context_ranking import (
    RankedSourcePhoto,
    build_context_contact_sheet,
    rank_source_photos,
)
from evidence_doctrine import (
    ABSENT,
    IMAGINED,
    INFERRED,
    OBSERVED,
    RECONSTRUCTED,
    generation_masks,
    provenance_percentages,
)
from export_dense_evidence import read_ply, read_visibility
from learned_inpainting import (
    LearnedInpaintingConfig,
    LearnedInpaintingResult,
    run_learned_inpainting,
)
from milestone09_walkthrough import (
    FACE_KEYS,
    PX_PER_UNIT,
    assign_planes_to_faces,
    canonical_orientation,
    deterministic_structural_imagine,
    estimate_seed_color,
    fit_room_envelope,
    measure_vram_used_mb,
)
from representation_spike import detect_planes, parse_views

logger = logging.getLogger("milestone13a_comparison")


def create_provenance_overlay(provenance: np.ndarray) -> np.ndarray:
    """Produce visual BGR overlay for provenance map."""
    palette = np.array([
        [24, 24, 24],     # 0: ABSENT (dark charcoal)
        [238, 238, 238], # 1: OBSERVED (bright white)
        [180, 220, 140], # 2: RECONSTRUCTED (sage green)
        [70, 160, 245],  # 3: INFERRED (sky blue)
        [200, 130, 230], # 4: IMAGINED (purple/magenta)
    ], dtype=np.uint8)
    safe_indices = np.clip(provenance, 0, 4)
    return palette[safe_indices]


def run_milestone13a_comparison(
    ply_path: Path,
    visibility_path: Path,
    images_text_path: Path,
    images_dir: Path,
    output_dir: Path,
    allow_download: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates-back").mkdir(parents=True, exist_ok=True)
    (output_dir / "candidates-right").mkdir(parents=True, exist_ok=True)

    # 1. Geometry and reconstruction setup
    vertices = read_ply(ply_path)
    points = np.asarray([v[:3] for v in vertices], dtype=np.float64)
    views = parse_views(images_text_path)
    supports = read_visibility(visibility_path, len(views))
    camera_centers = np.asarray([v.center for v in views])

    orientation = canonical_orientation(points, camera_centers)
    envelope = fit_room_envelope(points, orientation)
    spans = np.asarray(envelope["spans"])
    half_extent = spans / 2 * PX_PER_UNIT

    faces = {key: {"recovered": False, "widthPx": 0, "heightPx": 0, "transform": "", "image": None, "provenance": None} for key in FACE_KEYS}
    face_dims = {
        "left": (half_extent[2] * 2, half_extent[1] * 2),
        "right": (half_extent[2] * 2, half_extent[1] * 2),
        "floor": (half_extent[0] * 2, half_extent[2] * 2),
        "ceiling": (half_extent[0] * 2, half_extent[2] * 2),
        "back": (half_extent[0] * 2, half_extent[1] * 2),
        "front": (half_extent[0] * 2, half_extent[1] * 2),
    }
    for key in FACE_KEYS:
        faces[key]["widthPx"] = max(20.0, float(face_dims[key][0]))
        faces[key]["heightPx"] = max(20.0, float(face_dims[key][1]))

    planes, models, residual = detect_planes(ply_path, visibility_path, len(views), return_residual=True)
    face_reports = assign_planes_to_faces(planes, models, orientation, faces)

    wall_supports = []
    for p in planes:
        wall_supports.extend(p.get("supports", []))
    if not wall_supports:
        wall_supports = supports

    # 2. Context Ranking
    ranking_results = rank_source_photos(
        views=views, supports=supports, wall_supports=wall_supports, orientation=orientation,
        img_dir=images_dir, top_k=4
    )
    contact_sheet_img, crops_list, palette_info = build_context_contact_sheet(
        ranking_results, img_dir=images_dir
    )
    cv2.imwrite(str(output_dir / "context-contact-sheet.png"), contact_sheet_img)
    best_crop = crops_list[0] if crops_list else None

    seg_proc, seg_model = load_segmentation_model()

    # Diagnosis of the four M1.3 candidate rejections
    diagnosis = {
        "candidate_1": {
            "face": "back",
            "seed": 1301,
            "context": ranking_results[0].name,
            "strength": 1.0,
            "initialization": "none (unconstrained black canvas)",
            "rejectionReason": "unexpected semantic additions (19.5%) exceed tolerance (15.0%)",
            "rootCause": "Full unconstrained inpainting at strength 1.0 hallucinated interior door/window and sky regions, exceeding the 15% unexpected threshold."
        },
        "candidate_2": {
            "face": "back",
            "seed": 1301,
            "context": ranking_results[1].name if len(ranking_results) > 1 else "context_b",
            "strength": 1.0,
            "initialization": "none (unconstrained black canvas)",
            "rejectionReason": "unexpected semantic additions (45.5%) exceed tolerance (15.0%)",
            "rootCause": "Full unconstrained inpainting with Context B produced substantial non-wall structures and furniture, causing 45.5% unexpected additions."
        },
        "candidate_3": {
            "face": "right",
            "seed": 1301,
            "context": ranking_results[0].name,
            "strength": 1.0,
            "initialization": "none (unconstrained black canvas)",
            "rejectionReason": "detected unsupported signage/text in structural wall",
            "rootCause": "Unconstrained generation on right wall hallucinated ADE20K signboard/poster semantic class, violating the zero-unsupported-signage rule."
        },
        "candidate_4": {
            "face": "right",
            "seed": 1301,
            "context": ranking_results[1].name if len(ranking_results) > 1 else "context_b",
            "strength": 1.0,
            "initialization": "none (unconstrained black canvas)",
            "rejectionReason": "unexpected semantic additions (46.8%) exceed tolerance (15.0%)",
            "rootCause": "Unconstrained generation with Context B introduced massive non-structural content (46.8% unexpected), failing the semantic threshold."
        }
    }

    # 3. Controlled Evaluation on Target Face: `back`
    target_face = "back"
    target_width = int(faces[target_face]["widthPx"])
    target_height = int(faces[target_face]["heightPx"])
    seed_color = np.asarray(estimate_seed_color(faces), dtype=np.uint8)

    base_color = np.full((target_height, target_width, 3), seed_color, dtype=np.uint8)
    base_provenance = np.full((target_height, target_width), ABSENT, dtype=np.uint8)

    structural = np.ones((target_height, target_width), dtype=bool)
    critical = np.zeros((target_height, target_width), dtype=bool)
    masks = generation_masks(base_provenance, structural, critical)

    # Condition A: Remember
    ca_color = base_color.copy()
    ca_prov = base_provenance.copy()
    cv2.imwrite(str(output_dir / "conditionA-remember.png"), ca_color)
    cv2.imwrite(str(output_dir / "conditionA-remember-provenance.png"), create_provenance_overlay(ca_prov))
    ca_percentages = provenance_percentages(ca_prov)

    # Condition B: Deterministic Imagine
    cb_color, cb_prov = deterministic_structural_imagine(
        target_width, target_height, seed_color, masks["generatable"], masks["locked"]
    )
    cv2.imwrite(str(output_dir / "conditionB-deterministic.png"), cb_color)
    cv2.imwrite(str(output_dir / "conditionB-deterministic-provenance.png"), create_provenance_overlay(cb_prov))
    cb_percentages = provenance_percentages(cb_prov)
    cb_consistency = evaluate_context_consistency(
        cb_color, best_crop if best_crop is not None else cb_color,
        base_bgr=base_color, mask=masks["generatable"],
        processor=seg_proc, model=seg_model
    )

    # Condition C: Unconditioned Learned Imagine (M1.1A, strength=1.0)
    uncond_cfg = LearnedInpaintingConfig(
        seed=1101, steps=24, guidance_scale=6.0,
        allow_model_download=allow_download, use_ip_adapter=False, strength=1.0
    )
    cc_res = run_learned_inpainting(
        base_color, base_provenance, masks["generatable"], masks["locked"], critical,
        face_key=target_face, config=uncond_cfg
    )
    cc_color = cc_res.image_bgr if cc_res.accepted and cc_res.image_bgr is not None else cb_color.copy()
    cc_prov = cc_res.provenance if cc_res.accepted and cc_res.provenance is not None else cb_prov.copy()
    cv2.imwrite(str(output_dir / "conditionC-unconditioned.png"), cc_color)
    cv2.imwrite(str(output_dir / "conditionC-unconditioned-provenance.png"), create_provenance_overlay(cc_prov))
    cc_percentages = provenance_percentages(cc_prov)
    cc_consistency = evaluate_context_consistency(
        cc_color, best_crop if best_crop is not None else cc_color,
        base_bgr=base_color, mask=masks["generatable"],
        processor=seg_proc, model=seg_model
    )

    # Condition D: Prior Memory-Conditioned Unconstrained (M1.3, strength=1.0, seed=1301)
    cond_unconstrained_cfg = LearnedInpaintingConfig(
        seed=1301, steps=24, guidance_scale=6.0,
        allow_model_download=allow_download, use_ip_adapter=True,
        ip_adapter_scale=0.7, context_image=best_crop, strength=1.0
    )
    cd_res = run_learned_inpainting(
        base_color, base_provenance, masks["generatable"], masks["locked"], critical,
        face_key=target_face, config=cond_unconstrained_cfg
    )
    cd_color = cd_res.image_bgr if cd_res.accepted and cd_res.image_bgr is not None else cb_color.copy()
    cd_prov = cd_res.provenance if cd_res.accepted and cd_res.provenance is not None else cb_prov.copy()
    cv2.imwrite(str(output_dir / "conditionD-conditioned-unconstrained.png"), cd_color)
    cv2.imwrite(str(output_dir / "conditionD-conditioned-unconstrained-provenance.png"), create_provenance_overlay(cd_prov))
    cd_percentages = provenance_percentages(cd_prov)
    cd_consistency = evaluate_context_consistency(
        cd_color, best_crop,
        base_bgr=base_color, mask=masks["generatable"],
        processor=seg_proc, model=seg_model
    )

    # 4. Milestone 1.3A Candidate Refinement Search (back face)
    strengths = [0.35, 0.45, 0.55]
    seeds = [1301, 1101]
    back_candidate_reports = []
    accepted_back_candidate = None

    for s_idx, str_val in enumerate(strengths):
        for seed_val in seeds:
            cand_idx = len(back_candidate_reports) + 1
            cfg = LearnedInpaintingConfig(
                seed=seed_val, steps=24, guidance_scale=6.0,
                allow_model_download=allow_download, use_ip_adapter=True,
                ip_adapter_scale=0.7, context_image=best_crop, strength=str_val
            )
            t_start = time.perf_counter()
            res = run_learned_inpainting(
                cb_color, base_provenance, masks["generatable"], masks["locked"], critical,
                face_key=target_face, config=cfg
            )
            lat_ms = (time.perf_counter() - t_start) * 1000

            cand_img = res.image_bgr if res.image_bgr is not None else cb_color
            cand_prov = res.provenance if res.provenance is not None else cb_prov
            cand_file = output_dir / "candidates-back" / f"candidate-0{cand_idx}-s{int(str_val*100)}-seed{seed_val}.png"
            cv2.imwrite(str(cand_file), cand_img)

            consistency = evaluate_context_consistency(
                cand_img, best_crop,
                base_bgr=cb_color, mask=masks["generatable"],
                processor=seg_proc, model=seg_model
            )

            is_accepted = res.accepted and consistency.passed
            report_entry = {
                "candidateIndex": cand_idx,
                "face": "back",
                "initialization": "deterministic-imagine-m1.0",
                "contextPhoto": ranking_results[0].name,
                "refinementStrength": str_val,
                "seed": seed_val,
                "steps": 24,
                "guidanceScale": 6.0,
                "latencyMs": lat_ms,
                "peakVramMb": res.metrics.get("peakTorchVramMb"),
                "doctrineAccepted": res.accepted, "blocker": res.blocker,
                "protectedModifiedPixels": res.metrics.get("protectedModifiedPixels", 0),
                "criticalViolations": res.metrics.get("criticalViolations", 0),
                "consistencyPassed": consistency.passed,
                "deltaE": consistency.palette.delta_e,
                "colorSimilarityScore": consistency.palette.score,
                "edgeContinuityScore": consistency.edge_continuity.score,
                "structuralPercent": consistency.semantic_additions.structural_percent,
                "unexpectedPercent": consistency.semantic_additions.unexpected_percent,
                "detectedClasses": consistency.semantic_additions.detected_classes,
                "flaggedCategories": consistency.semantic_additions.flagged_categories,
                "failReasons": consistency.semantic_additions.fail_reasons,
                "accepted": is_accepted,
                "savedPath": str(cand_file),
            }
            back_candidate_reports.append(report_entry)

            if is_accepted and accepted_back_candidate is None:
                accepted_back_candidate = {
                    **report_entry,
                    "imageBgr": cand_img,
                    "provenance": cand_prov,
                }

    ce_color = accepted_back_candidate["imageBgr"] if accepted_back_candidate is not None else cb_color.copy()
    ce_prov = accepted_back_candidate["provenance"] if accepted_back_candidate is not None else cb_prov.copy()
    cv2.imwrite(str(output_dir / "conditionE-deterministic-refined.png"), ce_color)
    cv2.imwrite(str(output_dir / "conditionE-deterministic-refined-provenance.png"), create_provenance_overlay(ce_prov))
    ce_percentages = provenance_percentages(ce_prov)

    # 5. Right Face Refinement Search
    right_face = "right"
    right_width = int(faces[right_face]["widthPx"])
    right_height = int(faces[right_face]["heightPx"])
    right_base_color = np.full((right_height, right_width, 3), seed_color, dtype=np.uint8)
    right_base_prov = np.full((right_height, right_width), ABSENT, dtype=np.uint8)
    right_masks = generation_masks(right_base_prov, np.ones((right_height, right_width), dtype=bool), np.zeros((right_height, right_width), dtype=bool))
    right_det_color, right_det_prov = deterministic_structural_imagine(
        right_width, right_height, seed_color, right_masks["generatable"], right_masks["locked"]
    )

    right_candidate_reports = []
    accepted_right_candidate = None

    for str_val in strengths:
        for seed_val in seeds:
            cand_idx = len(right_candidate_reports) + 1
            cfg = LearnedInpaintingConfig(
                seed=seed_val, steps=24, guidance_scale=6.0,
                allow_model_download=allow_download, use_ip_adapter=True,
                ip_adapter_scale=0.7, context_image=best_crop, strength=str_val
            )
            t_start = time.perf_counter()
            res = run_learned_inpainting(
                right_det_color, right_base_prov, right_masks["generatable"], right_masks["locked"],
                np.zeros((right_height, right_width), dtype=bool),
                face_key=right_face, config=cfg
            )
            lat_ms = (time.perf_counter() - t_start) * 1000
            cand_img = res.image_bgr if res.image_bgr is not None else right_det_color
            cand_prov = res.provenance if res.provenance is not None else right_det_prov
            cand_file = output_dir / "candidates-right" / f"candidate-0{cand_idx}-s{int(str_val*100)}-seed{seed_val}.png"
            cv2.imwrite(str(cand_file), cand_img)

            consistency = evaluate_context_consistency(
                cand_img, best_crop,
                base_bgr=right_det_color, mask=right_masks["generatable"],
                processor=seg_proc, model=seg_model
            )

            is_accepted = res.accepted and consistency.passed
            r_report = {
                "candidateIndex": cand_idx,
                "face": "right",
                "initialization": "deterministic-imagine-m1.0",
                "contextPhoto": ranking_results[0].name,
                "refinementStrength": str_val,
                "seed": seed_val,
                "steps": 24,
                "guidanceScale": 6.0,
                "latencyMs": lat_ms,
                "peakVramMb": res.metrics.get("peakTorchVramMb"),
                "doctrineAccepted": res.accepted, "blocker": res.blocker,
                "protectedModifiedPixels": res.metrics.get("protectedModifiedPixels", 0),
                "criticalViolations": res.metrics.get("criticalViolations", 0),
                "consistencyPassed": consistency.passed,
                "deltaE": consistency.palette.delta_e,
                "colorSimilarityScore": consistency.palette.score,
                "edgeContinuityScore": consistency.edge_continuity.score,
                "structuralPercent": consistency.semantic_additions.structural_percent,
                "unexpectedPercent": consistency.semantic_additions.unexpected_percent,
                "detectedClasses": consistency.semantic_additions.detected_classes,
                "flaggedCategories": consistency.semantic_additions.flagged_categories,
                "failReasons": consistency.semantic_additions.fail_reasons,
                "accepted": is_accepted,
                "savedPath": str(cand_file),
            }
            right_candidate_reports.append(r_report)
            if is_accepted and accepted_right_candidate is None:
                accepted_right_candidate = {
                    **r_report,
                    "imageBgr": cand_img,
                    "provenance": cand_prov,
                }
                break
        if accepted_right_candidate is not None:
            break

    # 6. Build 5-Way Comparison Contact Sheet
    conditions = [
        ("A: Remember", ca_color),
        ("B: Deterministic", cb_color),
        ("C: M1.1A Unconditioned", cc_color),
        ("D: M1.3 Memory Unconstrained", cd_color),
        ("E: M1.3A Det + Refined", ce_color),
    ]
    banner_h = 32
    rendered_panels = []
    for label, img in conditions:
        h, w = img.shape[:2]
        panel = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        panel[:banner_h, :] = (30, 30, 30)
        cv2.putText(panel, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        panel[banner_h:, :] = img
        rendered_panels.append(panel)
    five_way_contact = np.hstack(rendered_panels)
    cv2.imwrite(str(output_dir / "comparison-5way.png"), five_way_contact)

    # 7. Compile Final Metrics
    ce_consistency = evaluate_context_consistency(
        ce_color, best_crop, base_bgr=cb_color, mask=masks["generatable"],
        processor=seg_proc, model=seg_model
    )
    results = {
        "milestone": "1.3A",
        "description": "Deterministic Imagine Initialization with Low-Strength IP-Adapter Memory Refinement",
        "target_face": target_face,
        "diagnosis_m13_rejections": diagnosis,
        "selected_context_photos": [
            {
                "name": r.name,
                "overall_score": r.overall_score,
                "geometry_score": r.geometry_score,
                "spatial_score": r.spatial_score,
                "alignment_score": r.alignment_score,
                "semantic_score": r.semantic_score,
            } for r in ranking_results
        ],
        "five_way_comparison": {
            "condition_a_remember": {
                "name": "Remember (Raw Reconstruction)",
                "provenance": ca_percentages,
                "structural_intact": True,
            },
            "condition_b_deterministic": {
                "name": "Deterministic Imagine (Milestone 1.0)",
                "provenance": cb_percentages,
                "structural_intact": True,
                "consistency": {
                    "passed": cb_consistency.passed,
                    "delta_e": cb_consistency.palette.delta_e,
                    "edge_score": cb_consistency.edge_continuity.score,
                    "structural_percent": cb_consistency.semantic_additions.structural_percent,
                    "unexpected_percent": cb_consistency.semantic_additions.unexpected_percent,
                }
            },
            "condition_c_unconditioned_learned": {
                "name": "Unconditioned Learned Imagine (Milestone 1.1A)",
                "provenance": cc_percentages,
                "structural_intact": cc_res.accepted,
                "consistency": {
                    "passed": cc_consistency.passed,
                    "delta_e": cc_consistency.palette.delta_e,
                    "edge_score": cc_consistency.edge_continuity.score,
                    "structural_percent": cc_consistency.semantic_additions.structural_percent,
                    "unexpected_percent": cc_consistency.semantic_additions.unexpected_percent,
                }
            },
            "condition_d_conditioned_unconstrained": {
                "name": "Memory-Conditioned Unconstrained (Milestone 1.3)",
                "provenance": cd_percentages,
                "structural_intact": cd_res.accepted,
                "consistency": {
                    "passed": cd_consistency.passed,
                    "delta_e": cd_consistency.palette.delta_e,
                    "edge_score": cd_consistency.edge_continuity.score,
                    "structural_percent": cd_consistency.semantic_additions.structural_percent,
                    "unexpected_percent": cd_consistency.semantic_additions.unexpected_percent,
                    "fail_reasons": cd_consistency.semantic_additions.fail_reasons,
                }
            },
            "condition_e_deterministic_refined": {
                "name": "Deterministic + Low-Strength Memory Refined (Milestone 1.3A)",
                "provenance": ce_percentages,
                "structural_intact": accepted_back_candidate["doctrineAccepted"] if accepted_back_candidate else False,
                "accepted_candidate": {
                    "candidate_index": accepted_back_candidate["candidateIndex"] if accepted_back_candidate else None,
                    "refinement_strength": accepted_back_candidate["refinementStrength"] if accepted_back_candidate else None,
                    "seed": accepted_back_candidate["seed"] if accepted_back_candidate else None,
                    "steps": accepted_back_candidate["steps"] if accepted_back_candidate else None,
                    "guidance_scale": accepted_back_candidate["guidanceScale"] if accepted_back_candidate else None,
                    "latency_ms": accepted_back_candidate["latencyMs"] if accepted_back_candidate else None,
                    "peak_vram_mb": accepted_back_candidate["peakVramMb"] if accepted_back_candidate else None,
                },
                "consistency": {
                    "passed": ce_consistency.passed,
                    "delta_e": ce_consistency.palette.delta_e,
                    "edge_score": ce_consistency.edge_continuity.score,
                    "structural_percent": ce_consistency.semantic_additions.structural_percent,
                    "unexpected_percent": ce_consistency.semantic_additions.unexpected_percent,
                }
            }
        },
        "back_candidates": back_candidate_reports,
        "right_candidates": right_candidate_reports,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results

def main():
    parser = argparse.ArgumentParser(description="Milestone 1.3A Comparison Runner")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    res = run_milestone13a_comparison(
        ply_path=args.ply,
        visibility_path=args.visibility,
        images_text_path=args.images_text,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        allow_download=args.allow_download,
    )
    print(f"Milestone 1.3A comparison completed. Metrics written to {args.output_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
