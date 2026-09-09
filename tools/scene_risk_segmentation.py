"""Milestone 0.8, Phase 2: scene risk segmentation.

Classifies every pixel of each registered source photograph into one of
three risk tiers using the same locally cached ADE20K semantic
segmentation model already validated in Milestones 0.2-0.3
(`openmmlab/upernet-convnext-tiny`, no cloud calls):

  structural  -- wall/floor/ceiling/door/window/stairs/etc: low risk,
                 conservative inference is allowed under Doctrine v2.
  object      -- ordinary furniture/appliances: medium risk. Photographed
                 surfaces must dominate; this script only classifies, it
                 does not decide how much (if any) shape completion is
                 appropriate (see Phase 8's object_completion_test.py).
  critical    -- people, faces, pets, artwork, signage, text, mirrors,
                 personal objects: high risk. Never inferred over.

This script itself contains no photos or identifying data. It writes
private per-image mask visualizations and a repository-safe aggregate
summary (percentages only, no filenames retained in the aggregate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_doctrine import risk_masks_from_labels


def segment_image(image_bgr: np.ndarray, processor, model, torch) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    labels = processor.post_process_semantic_segmentation(
        type("Output", (), {"logits": logits})(),
        target_sizes=[image_bgr.shape[:2]],
    )[0].cpu().numpy()
    return labels


def visualize(masks: dict, shape: tuple[int, int]) -> np.ndarray:
    vis = np.zeros((*shape, 3), dtype=np.uint8)
    vis[masks["structural"]] = (60, 200, 60)     # green: low-risk, inferable
    vis[masks["object"]] = (200, 160, 40)        # blue-ish: medium-risk
    vis[masks["critical"]] = (40, 40, 220)        # red: never inferred
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Private source-photo directory")
    parser.add_argument("--output", required=True, help="Private output directory")
    parser.add_argument("--model", default="openmmlab/upernet-convnext-tiny")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    images_dir = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model)
    model.eval()

    per_image = []
    label_id_to_name = model.config.id2label
    class_pixel_totals: dict[str, int] = {}

    for image_path in sorted(images_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        labels = segment_image(image, processor, model, torch)
        masks = risk_masks_from_labels(labels, label_id_to_name)

        total = labels.size
        entry = {
            "name": image_path.name,
            "structuralPercent": float(masks["structural"].mean() * 100),
            "objectPercent": float(masks["object"].mean() * 100),
            "criticalPercent": float(masks["critical"].mean() * 100),
        }
        per_image.append(entry)

        for class_id in np.unique(labels):
            name = label_id_to_name.get(int(class_id), str(class_id))
            class_pixel_totals[name] = class_pixel_totals.get(name, 0) + int((labels == class_id).sum())

        vis = visualize(masks, labels.shape)
        overlay = cv2.addWeighted(image, 0.55, vis, 0.45, 0)
        cv2.imwrite(str(out_dir / f"risk-{image_path.name}.png"), overlay)
        np.savez_compressed(
            out_dir / f"risk-masks-{image_path.stem}.npz",
            structural=masks["structural"],
            object=masks["object"],
            critical=masks["critical"],
        )

    total_pixels = sum(class_pixel_totals.values())
    class_breakdown = {
        name: round(count / total_pixels * 100, 3)
        for name, count in sorted(class_pixel_totals.items(), key=lambda kv: -kv[1])
    }

    aggregate = {
        "model": args.model,
        "imageCount": len(per_image),
        "meanStructuralPercent": float(np.mean([e["structuralPercent"] for e in per_image])) if per_image else 0.0,
        "meanObjectPercent": float(np.mean([e["objectPercent"] for e in per_image])) if per_image else 0.0,
        "meanCriticalPercent": float(np.mean([e["criticalPercent"] for e in per_image])) if per_image else 0.0,
        "classBreakdownPercent": class_breakdown,
    }

    (out_dir / "per-image.json").write_text(json.dumps(per_image, indent=2))
    (out_dir / "aggregate-summary.json").write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
