#!/usr/bin/env python3
"""Milestone 0.6 Phase 1-2: spatial viewpoint graph over the Clean 13-photo set.

Builds a graph where each registered photograph is a node (recovered camera
pose) and edges exist only where evidence supports a spatial relationship:
shared SfM landmarks, camera distance/angle, and a camera-geometry-only
"support gate continuity" score sampled along the straight path between the
two camera centers (same gate design validated in Milestone 0.5 -- distance
and viewing-angle to the nearest OTHER training camera, never renderer
opacity).

Every edge is classified into:
  A. strong  -- support stays high along the whole path; a local radiance
                bridge is credible.
  B. weak    -- some shared evidence, but support drops mid-path; only a
                short partial transition before handing off to the next
                captured photo is credible.
  C. none    -- not connected by direct evidence; no edge is created.

Output: a private graph.json (includes recovered camera poses -- identifying
metadata -- so it must stay outside Git) and a repo-safe aggregate summary
with only counts/statistics, no poses or filenames.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import parse_camera, parse_views, angle_degrees, rank_pairs


def support_score_at(point: np.ndarray, other_centers: np.ndarray, other_forwards: np.ndarray, max_distance: float, max_angle_deg: float = 45.0) -> float:
    """Camera-geometry-only support score for one 3D point (no rendered opacity)."""
    best = 0.0
    for center, forward in zip(other_centers, other_forwards):
        to_point = point - center
        distance = float(np.linalg.norm(to_point))
        if distance < 1e-9:
            continue
        direction = to_point / distance
        cos_angle = np.clip(float(np.dot(direction, forward)), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        distance_score = np.clip(1.0 - distance / max_distance, 0.0, 1.0)
        angle_score = np.clip(1.0 - angle_deg / max_angle_deg, 0.0, 1.0)
        best = max(best, distance_score * angle_score)
    return best


def classify_edge(pair: dict, path_scores: list[float]) -> str:
    if pair["shared"] < 8:
        return "none"
    min_support = min(path_scores)
    mean_support = float(np.mean(path_scores))
    if min_support >= 0.45 and mean_support >= 0.6:
        return "strong"
    if mean_support >= 0.25:
        return "weak"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-summary", type=Path, required=True)
    parser.add_argument("--path-samples", type=int, default=9)
    args = parser.parse_args()

    parse_camera(args.camera_text)  # validates PINHOLE model; camera itself unused here.
    views = parse_views(args.images_text)
    centers = np.stack([view.center for view in views])
    forwards = np.stack([view.forward for view in views])
    coverage_radius = float(np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
    max_distance = max(coverage_radius * 2.0, 1e-3)

    pairs = rank_pairs(views)
    name_to_index = {view.name: index for index, view in enumerate(views)}

    edges = []
    for pair in pairs:
        first, second = pair["first"], pair["second"]
        first_index, second_index = name_to_index[first.name], name_to_index[second.name]
        other_mask = np.ones(len(views), dtype=bool)
        other_mask[[first_index, second_index]] = False
        other_centers = centers[other_mask]
        other_forwards = forwards[other_mask]

        path_scores = []
        for fraction in np.linspace(0.0, 1.0, args.path_samples):
            point = first.center * (1 - fraction) + second.center * fraction
            path_scores.append(support_score_at(point, other_centers, other_forwards, max_distance))
        # The endpoints themselves are always evidence-anchored (real cameras stood there).
        path_scores[0] = max(path_scores[0], 1.0)
        path_scores[-1] = max(path_scores[-1], 1.0)

        classification = classify_edge(pair, path_scores)
        if classification == "none":
            continue
        edges.append({
            "from": first.name,
            "to": second.name,
            "sharedLandmarks": pair["shared"],
            "cameraDistance": pair["distance"],
            "angularDifferenceDegrees": pair["angle"],
            "pathSupportScores": [round(float(s), 4) for s in path_scores],
            "minPathSupport": round(float(min(path_scores)), 4),
            "meanPathSupport": round(float(np.mean(path_scores)), 4),
            "classification": classification,
        })

    nodes = [
        {"name": view.name, "center": view.center.tolist(), "quaternion": view.quaternion.tolist(), "forward": view.forward.tolist()}
        for view in views
    ]

    connected_names = set()
    for edge in edges:
        connected_names.add(edge["from"])
        connected_names.add(edge["to"])
    isolated = [view.name for view in views if view.name not in connected_names]

    graph = {
        "nodes": nodes,
        "edges": edges,
        "isolatedNodes": isolated,
        "maxDistanceUsedForSupportGate": max_distance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    strong = [e for e in edges if e["classification"] == "strong"]
    weak = [e for e in edges if e["classification"] == "weak"]
    summary = {
        "totalRegisteredNodes": len(views),
        "connectedNodes": len(connected_names),
        "isolatedNodes": len(isolated),
        "totalEdges": len(edges),
        "strongEdges": len(strong),
        "weakEdges": len(weak),
        "averageSharedLandmarksPerEdge": round(float(np.mean([e["sharedLandmarks"] for e in edges])), 2) if edges else 0,
        "averageMinPathSupportStrongEdges": round(float(np.mean([e["minPathSupport"] for e in strong])), 4) if strong else None,
        "averageMinPathSupportWeakEdges": round(float(np.mean([e["minPathSupport"] for e in weak])), 4) if weak else None,
        "method": "Edges use shared SfM landmarks, camera distance/angle, and a camera-geometry-only support-gate score sampled along the straight path between camera centers (independent of renderer opacity). Nodes/poses omitted here as identifying metadata; see private graph.json.",
    }
    args.repo_summary.parent.mkdir(parents=True, exist_ok=True)
    args.repo_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote private graph to {args.output}")
    print(f"Wrote repo-safe summary to {args.repo_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
