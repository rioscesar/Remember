#!/usr/bin/env python3
"""Milestone 0.8, Phase 8: ordinary-object completion test (couch).

Doctrine v2 explicitly scopes this milestone's object handling to
classification only -- this script does NOT attempt 3D mesh completion of
furniture. Instead it produces an honest 2D evidence-coverage proxy:

  1. Segment every source photo (same cached ADE20K model as Phase 2) and
     find the two photos with the strongest "sofa" pixel coverage.
  2. Match SIFT features between the two photos and estimate a planar
     homography (this is an approximation -- a couch is not a plane -- so
     the result is reported as a coverage *proxy*, not a claim of accurate
     3D shape recovery).
  3. Warp photo A's couch mask into photo B's frame and union it with
     photo B's own couch mask, then compare that combined-evidence
     coverage against the best single-view coverage alone.

This answers a narrower, honest question: does combining two ordinary
views of the couch increase how much of it is directly evidenced, without
claiming to reconstruct its true 3D shape. No pixels are fabricated by
this script; it only measures overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_transition_prototype import match_features


def segment(image_bgr: np.ndarray, processor, model, torch) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = processor.post_process_semantic_segmentation(
        type("Output", (), {"logits": logits})(),
        target_sizes=[image_bgr.shape[:2]],
    )[0].cpu().numpy()
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", default="openmmlab/upernet-convnext-tiny")
    parser.add_argument("--target-label", default="sofa")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    args.output.mkdir(parents=True, exist_ok=True)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval()
    id2label = model.config.id2label
    target_ids = {i for i, name in id2label.items() if args.target_label in name.lower()}
    if not target_ids:
        raise RuntimeError(f"No ADE20K label matched '{args.target_label}'.")

    candidates = []
    for image_path in sorted(args.images.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        labels = segment(image, processor, model, torch)
        mask = np.isin(labels, list(target_ids))
        candidates.append({"path": image_path, "image": image, "mask": mask, "pixels": int(mask.sum())})

    candidates.sort(key=lambda c: -c["pixels"])
    if len(candidates) < 2 or candidates[1]["pixels"] < 2000:
        raise RuntimeError(f"Fewer than two photos show enough of '{args.target_label}' to compare.")

    a, b = candidates[0], candidates[1]
    match = match_features(a["image"], b["image"])
    result = {
        "targetLabel": args.target_label,
        "photoA": a["path"].name,
        "photoB": b["path"].name,
        "photoAMaskPixels": a["pixels"],
        "photoBMaskPixels": b["pixels"],
    }

    if match is None:
        result["matched"] = False
        result["note"] = "Insufficient feature matches between the two strongest views; no combined-coverage claim can be made."
    else:
        homography = match["homography"]
        warped_a_mask = cv2.warpPerspective(
            a["mask"].astype(np.uint8), homography, (b["image"].shape[1], b["image"].shape[0])
        ) > 0
        union_mask = warped_a_mask | b["mask"]
        frame_area = b["mask"].size
        best_single_coverage = max(a["pixels"], b["pixels"]) / frame_area * 100
        combined_coverage = int(union_mask.sum()) / frame_area * 100
        newly_evidenced = int((union_mask & ~b["mask"]).sum()) / frame_area * 100
        result.update({
            "matched": True,
            "inlierMatchCount": match["n_inliers"],
            "bestSingleViewCoveragePercentOfFrameB": best_single_coverage,
            "combinedTwoViewCoveragePercentOfFrameB": combined_coverage,
            "newlyEvidencedByPhotoAPercentOfFrameB": newly_evidenced,
            "caveat": "Planar homography is an approximation for a non-planar couch; treat as a 2D evidence-coverage proxy, not a 3D shape reconstruction.",
        })

        overlay_b = b["image"].copy()
        overlay_b[b["mask"]] = (0.6 * overlay_b[b["mask"]] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
        overlay_b[warped_a_mask & ~b["mask"]] = (
            0.6 * overlay_b[warped_a_mask & ~b["mask"]] + 0.4 * np.array([0, 165, 255])
        ).astype(np.uint8)
        cv2.imwrite(str(args.output / "couch-combined-coverage.png"), overlay_b)

    (args.output / "object-completion-metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
