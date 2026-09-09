#!/usr/bin/env python3
"""Milestone 0.6: render a private top-down visualization of the spatial
viewpoint graph (camera nodes projected to their best-fit floor plane,
edges colored by classification). Founder-review artifact only; contains
camera positions derived from private photographs, so the PNG itself must
never be committed -- only this generation script and aggregate metrics are
repo-safe.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    centers = np.array([node["center"] for node in graph["nodes"]])
    names = [node["name"] for node in graph["nodes"]]

    centroid = centers.mean(axis=0)
    _, _, axes = np.linalg.svd(centers - centroid, full_matrices=False)
    # Use the two directions of greatest camera spread as the floor-plane basis
    # (cameras mostly move horizontally, so this approximates a top-down map).
    basis = axes[:2]
    flat = (centers - centroid) @ basis.T

    margin = 80
    span = flat.max(axis=0) - flat.min(axis=0)
    span[span < 1e-6] = 1.0
    scale = (args.size - 2 * margin) / span.max()
    pixels = (flat - flat.min(axis=0)) * scale + margin
    pixels[:, 1] = args.size - pixels[:, 1]

    canvas = np.full((args.size, args.size, 3), 24, dtype=np.uint8)
    name_to_pixel = {name: tuple(pixels[i].astype(int)) for i, name in enumerate(names)}

    color_by_class = {"strong": (80, 220, 80), "weak": (60, 140, 220)}
    for edge in graph["edges"]:
        p1 = name_to_pixel[edge["from"]]
        p2 = name_to_pixel[edge["to"]]
        color = color_by_class.get(edge["classification"], (90, 90, 90))
        thickness = 3 if edge["classification"] == "strong" else 1
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    isolated = set(graph.get("isolatedNodes", []))
    for name, (x, y) in name_to_pixel.items():
        color = (60, 60, 220) if name in isolated else (230, 230, 230)
        cv2.circle(canvas, (x, y), 10, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 10, (10, 10, 10), 2, cv2.LINE_AA)

    legend_y = 30
    for label, color in [("strong bridge", (80, 220, 80)), ("weak bridge", (60, 140, 220)), ("camera node", (230, 230, 230))]:
        cv2.circle(canvas, (30, legend_y), 8, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (48, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        legend_y += 26

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"Wrote graph visualization to {args.output}")


if __name__ == "__main__":
    main()
