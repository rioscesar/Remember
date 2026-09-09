#!/usr/bin/env python3
"""Milestone 0.6 Phase 3-5: render the one evidence-supported local radiance
bridge, gated per-pixel by the Milestone 0.5 camera-geometry support model.

Reuses the regularized checkpoint trained in Milestone 0.5
(radiance_spike_sparse.py) -- no new training happens here. For the single
edge the graph classified "strong", renders N interpolated frames between
the two real camera poses, and for each frame computes the same
opacity-independent support gate as Milestone 0.5 to fade any pixel that
falls outside evidence coverage.

Weak edges are deliberately NOT rendered with reconstruction: per the
founder brief, a weak bridge should hand off to the next captured
photograph rather than trust novel-view synthesis, so the navigation
prototype crossfades directly between the two real photographs instead.

Outputs private per-frame RGB, alpha, and support PNGs plus a manifest.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from representation_spike import Camera, View, parse_camera, parse_views
from radiance_spike import quaternion_rotation_torch, slerp, view_to_matrices
from radiance_spike_sparse import render_rgbd, compute_world_points, support_gate, fade_alpha


def load_checkpoint(path: Path, device: str) -> dict:
    raw = torch.load(path, map_location=device)
    return {name: torch.nn.Parameter(value.to(device)) for name, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    source_camera = parse_camera(args.camera_text)
    views = {view.name: view for view in parse_views(args.images_text)}
    scale = 1.0 / args.downsample
    render_camera = Camera(
        int(round(source_camera.width * scale)), int(round(source_camera.height * scale)),
        source_camera.fx * scale, source_camera.fy * scale, source_camera.cx * scale, source_camera.cy * scale,
    )

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    strong_edges = [e for e in graph["edges"] if e["classification"] == "strong"]
    if not strong_edges:
        raise RuntimeError("No strong edges in the graph; nothing to bridge.")

    all_centers = np.stack([np.array(n["center"]) for n in graph["nodes"]])
    all_forwards = np.stack([np.array(n["forward"]) for n in graph["nodes"]])
    coverage_radius = float(np.linalg.norm(all_centers - all_centers.mean(axis=0), axis=1).max())
    max_distance = max(coverage_radius * 2.0, 1e-3)

    params = load_checkpoint(args.checkpoint, device)

    manifest = {"edges": []}
    for edge in strong_edges:
        first, second = views[edge["from"]], views[edge["to"]]
        edge_dir = args.output / f"{edge['from']}__{edge['to']}"
        edge_dir.mkdir(exist_ok=True)
        frame_reports = []
        for index, fraction in enumerate(np.linspace(0.0, 1.0, args.frames)):
            center = first.center * (1 - fraction) + second.center * fraction
            quat = slerp(first.quaternion, second.quaternion, float(fraction))
            rotation = quaternion_rotation_torch(torch.tensor(quat, dtype=torch.float32)).numpy()
            interpolated = View(f"frame-{index}", rotation, center, quat, {})
            viewmat_np = np.eye(4, dtype=np.float32)
            viewmat_np[:3, :3] = interpolated.rotation
            viewmat_np[:3, 3] = interpolated.rotation @ (-interpolated.center)
            K_np = np.array([[render_camera.fx, 0, render_camera.cx], [0, render_camera.fy, render_camera.cy], [0, 0, 1]], dtype=np.float32)
            viewmat = torch.tensor(viewmat_np, device=device)
            K = torch.tensor(K_np, device=device)
            with torch.no_grad():
                rgb, depth, alpha, _ = render_rgbd(params, viewmat, K, render_camera.width, render_camera.height)
            rgb_np = rgb.cpu().numpy()
            alpha_np = alpha.cpu().numpy()
            depth_np = depth.cpu().numpy()
            world_points = compute_world_points(depth_np, viewmat_np, K_np)
            support_score, labels = support_gate(world_points, center, all_centers, all_forwards, max_distance)
            gated_alpha = fade_alpha(alpha_np, support_score)
            gated_rgb = rgb_np * gated_alpha[..., None]

            cv2.imwrite(str(edge_dir / f"frame-{index:02d}-rgb.png"), cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(edge_dir / f"frame-{index:02d}-gated-rgba.png"), cv2.cvtColor(
                np.dstack([(gated_rgb * 255).astype(np.uint8), (gated_alpha * 255).astype(np.uint8)]),
                cv2.COLOR_BGRA2RGBA,
            ))
            frame_reports.append({
                "frame": index,
                "fraction": float(fraction),
                "meanSupportScore": float(support_score.mean()),
                "supportedPercent": float((labels == 2).mean() * 100),
                "weaklySupportedPercent": float((labels == 1).mean() * 100),
                "unsupportedPercent": float((labels == 0).mean() * 100),
                "meanGatedAlpha": float(gated_alpha.mean()),
            })
        manifest["edges"].append({"from": edge["from"], "to": edge["to"], "frames": frame_reports})

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote bridge frames to {args.output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
