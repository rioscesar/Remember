#!/usr/bin/env python3
"""Create local-only image-based representation comparisons from a COLMAP workspace."""

import argparse
import html
import json
import math
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_dense_evidence import read_ply, read_visibility


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class View:
    name: str
    rotation: np.ndarray
    center: np.ndarray
    quaternion: np.ndarray
    observations: dict[int, tuple[float, float]]

    @property
    def forward(self) -> np.ndarray:
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])


def quaternion_rotation(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def parse_camera(path: Path) -> Camera:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].isdigit():
            _, model, width, height, *params = fields
            if model != "PINHOLE":
                raise RuntimeError(f"Expected an undistorted PINHOLE camera, found {model}.")
            fx, fy, cx, cy = map(float, params)
            return Camera(int(width), int(height), fx, fy, cx, cy)
    raise RuntimeError("No camera found.")


def parse_views(path: Path) -> list[View]:
    lines = path.read_text(encoding="utf-8").splitlines()
    views = []
    index = 0
    while index < len(lines):
        fields = lines[index].split()
        if len(fields) >= 10 and fields[0].isdigit() and fields[8].isdigit():
            q = np.array([float(value) for value in fields[1:5]])
            translation = np.array([float(value) for value in fields[5:8]])
            rotation = quaternion_rotation(q)
            center = -rotation.T @ translation
            observations = {}
            if index + 1 < len(lines):
                points = lines[index + 1].split()
                for offset in range(0, len(points) - 2, 3):
                    point_id = int(points[offset + 2])
                    if point_id >= 0:
                        observations[point_id] = (float(points[offset]), float(points[offset + 1]))
            views.append(View(" ".join(fields[9:]), rotation, center, q, observations))
            index += 2
        else:
            index += 1
    return sorted(views, key=lambda view: view.name)


def read_depth(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = b""
        while header.count(b"&") < 3:
            byte = file.read(1)
            if not byte:
                raise RuntimeError(f"Truncated depth-map header: {path}")
            header += byte
        width, height, channels = (int(value) for value in header.decode().split("&")[:3])
        values = np.frombuffer(file.read(width * height * channels * 4), dtype="<f4")
        if values.size != width * height * channels:
            raise RuntimeError(f"Truncated depth map: {path}")
        return values.reshape((height, width, channels)).squeeze()


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)), -1, 1)
    return math.degrees(math.acos(cosine))


def rank_pairs(views: list[View]) -> list[dict]:
    pairs = []
    for index, first in enumerate(views):
        for second in views[index + 1:]:
            shared = len(first.observations.keys() & second.observations.keys())
            if shared == 0:
                continue
            distance = float(np.linalg.norm(first.center - second.center))
            angle = angle_degrees(first.forward, second.forward)
            score = shared / (1 + angle / 30) / (1 + distance)
            pairs.append({
                "first": first,
                "second": second,
                "shared": shared,
                "distance": distance,
                "angle": angle,
                "score": score,
            })
    return sorted(pairs, key=lambda pair: pair["score"], reverse=True)


def detect_planes(
    ply: Path,
    visibility: Path,
    registered_views: int,
) -> tuple[list[dict], list[dict]]:
    vertices = read_ply(ply)
    visible_from = read_visibility(visibility, registered_views)
    selected = [
        (vertex[:3], views)
        for vertex, views in zip(vertices, visible_from)
        if len(views) >= 3
    ]
    points = np.asarray([item[0] for item in selected], dtype=np.float64)
    supports = [item[1] for item in selected]
    spans = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
    threshold = float(np.linalg.norm(spans) * 0.005)
    remaining = np.arange(len(points))
    rng = np.random.default_rng(20260908)
    planes = []
    models = []
    for plane_index in range(5):
        if remaining.size < 300:
            break
        candidates = points[remaining]
        best = np.array([], dtype=np.int64)
        for _ in range(750):
            sample = candidates[rng.choice(candidates.shape[0], 3, replace=False)]
            normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
            length = np.linalg.norm(normal)
            if length < 1e-9:
                continue
            normal /= length
            distances = np.abs((candidates - sample[0]) @ normal)
            inliers = np.flatnonzero(distances <= threshold)
            if inliers.size > best.size:
                best = inliers
        if best.size < 300:
            break
        global_inliers = remaining[best]
        inlier_points = points[global_inliers]
        centroid = inlier_points.mean(axis=0)
        _, _, axes = np.linalg.svd(inlier_points - centroid, full_matrices=False)
        normal = axes[2]
        distances = np.abs((inlier_points - centroid) @ normal)
        projected = (inlier_points - centroid) @ axes[:2].T
        extents = np.percentile(projected, 95, axis=0) - np.percentile(projected, 5, axis=0)
        source_views = set().union(*(supports[index] for index in global_inliers))
        planes.append({
            "plane": plane_index + 1,
            "multiViewPoints": int(best.size),
            "distinctSourceViews": len(source_views),
            "normal": normal.tolist(),
            "robustExtents": extents.tolist(),
            "distanceThreshold": threshold,
            "rmsResidual": float(np.sqrt(np.mean(distances * distances))),
        })
        models.append({
            "points": inlier_points,
            "supports": [supports[index] for index in global_inliers],
        })
        remaining = np.delete(remaining, best)
    return planes, models


def project_world(points: np.ndarray, view: View, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    camera_points = view.rotation @ (points - view.center).T
    valid = camera_points[2] > 0.01
    projected = np.column_stack((
        camera.fx * camera_points[0] / camera_points[2] + camera.cx,
        camera.fy * camera_points[1] / camera_points[2] + camera.cy,
    ))
    return projected, valid


def create_planar_proxy(
    model: dict,
    views: list[View],
    mvs_names: list[str],
    camera: Camera,
    images: Path,
    output: Path,
    target_pair: dict,
) -> dict:
    points = model["points"]
    supports = model["supports"]
    mvs_index = {name: index for index, name in enumerate(mvs_names)}
    source_counts = {
        view.name: sum(mvs_index.get(view.name) in point_support for point_support in supports)
        for view in views
    }
    source = max(views, key=lambda view: source_counts[view.name])
    target = View(
        "virtual-midpoint",
        quaternion_rotation(slerp(target_pair["first"].quaternion, target_pair["second"].quaternion, 0.5)),
        (target_pair["first"].center + target_pair["second"].center) / 2,
        np.zeros(4),
        {},
    )
    source_pixels, source_front = project_world(points, source, camera)
    target_pixels, target_front = project_world(points, target, camera)
    source_id = mvs_index[source.name]
    observed = np.array([source_id in support for support in supports])
    inside = (
        observed & source_front & target_front
        & (source_pixels[:, 0] >= 0) & (source_pixels[:, 0] < camera.width)
        & (source_pixels[:, 1] >= 0) & (source_pixels[:, 1] < camera.height)
        & (target_pixels[:, 0] >= 0) & (target_pixels[:, 0] < camera.width)
        & (target_pixels[:, 1] >= 0) & (target_pixels[:, 1] < camera.height)
    )
    source_points = source_pixels[inside].astype(np.float32)
    target_points = target_pixels[inside].astype(np.float32)
    if len(source_points) < 20:
        raise RuntimeError("The strongest plane has too little shared target-view support.")
    homography, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, 2.0)
    if homography is None:
        raise RuntimeError("Could not estimate an evidence-backed planar homography.")
    accepted = inliers.ravel().astype(bool)
    hull = cv2.convexHull(source_points[accepted].reshape(-1, 1, 2)).astype(np.int32)
    source_mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, hull, 255)
    source_image = cv2.imread(str(images / source.name), cv2.IMREAD_COLOR)
    warped = cv2.warpPerspective(source_image, homography, (camera.width, camera.height))
    warped_mask = cv2.warpPerspective(
        source_mask,
        homography,
        (camera.width, camera.height),
        flags=cv2.INTER_NEAREST,
    )
    write_supported_png(output / "planar-proxy.png", warped, warped_mask > 0)
    predicted = cv2.perspectiveTransform(source_points[accepted, None, :], homography)[:, 0]
    residual = np.linalg.norm(predicted - target_points[accepted], axis=1)
    return {
        "source": source.name,
        "targetBetween": [target_pair["first"].name, target_pair["second"].name],
        "supportingPlanePoints": int(inside.sum()),
        "homographyInliers": int(accepted.sum()),
        "homographyInlierPercent": float(accepted.mean() * 100),
        "rmsReprojectionPixels": float(np.sqrt(np.mean(residual * residual))),
        "boundedTargetCoveragePercent": float((warped_mask > 0).mean() * 100),
        "boundary": "convex hull of source-view observations supporting the recovered plane",
    }


def slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    second = second.copy()
    cosine = float(np.dot(first, second))
    if cosine < 0:
        second = -second
        cosine = -cosine
    if cosine > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    theta = math.acos(np.clip(cosine, -1, 1))
    return (
        math.sin((1 - fraction) * theta) / math.sin(theta) * first
        + math.sin(fraction * theta) / math.sin(theta) * second
    )


def project_source(
    image: np.ndarray,
    depth: np.ndarray,
    source: View,
    target_rotation: np.ndarray,
    target_center: np.ndarray,
    camera: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    valid_y, valid_x = np.nonzero(np.isfinite(depth) & (depth > 0))
    z = depth[valid_y, valid_x].astype(np.float64)
    camera_points = np.vstack((
        (valid_x - camera.cx) / camera.fx * z,
        (valid_y - camera.cy) / camera.fy * z,
        z,
    ))
    world = source.rotation.T @ camera_points + source.center[:, None]
    target = target_rotation @ (world - target_center[:, None])
    in_front = target[2] > 0.01
    target = target[:, in_front]
    colours = image[valid_y[in_front], valid_x[in_front]]
    screen_x = np.rint(camera.fx * target[0] / target[2] + camera.cx).astype(np.int32)
    screen_y = np.rint(camera.fy * target[1] / target[2] + camera.cy).astype(np.int32)
    inside = (
        (screen_x >= 0) & (screen_x < camera.width)
        & (screen_y >= 0) & (screen_y < camera.height)
    )
    screen_x, screen_y = screen_x[inside], screen_y[inside]
    target_depth = target[2, inside].astype(np.float32)
    colours = colours[inside]

    order = np.argsort(target_depth)[::-1]
    result = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    z_buffer = np.full((camera.height, camera.width), np.inf, dtype=np.float32)
    result[screen_y[order], screen_x[order]] = colours[order]
    z_buffer[screen_y[order], screen_x[order]] = target_depth[order]
    return result, z_buffer


def write_supported_png(path: Path, colour: np.ndarray, supported: np.ndarray) -> None:
    rgba = cv2.cvtColor(colour, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = np.where(supported, 255, 0).astype(np.uint8)
    cv2.imwrite(str(path), rgba)


def create_midpoint(
    pair: dict,
    camera: Camera,
    images: Path,
    depths: Path,
    output: Path,
    label: str,
) -> dict:
    first, second = pair["first"], pair["second"]
    target_center = (first.center + second.center) / 2
    target_rotation = quaternion_rotation(slerp(first.quaternion, second.quaternion, 0.5))
    rendered = []
    for view in (first, second):
        image = cv2.imread(str(images / view.name), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {view.name}.")
        depth = read_depth(depths / f"{view.name}.geometric.bin")
        if image.shape[:2] != depth.shape:
            raise RuntimeError(f"Image/depth dimensions disagree for {view.name}.")
        rendered.append(project_source(image, depth, view, target_rotation, target_center, camera))

    (first_colour, first_depth), (second_colour, second_depth) = rendered
    first_support = np.isfinite(first_depth)
    second_support = np.isfinite(second_depth)
    either = first_support | second_support
    both = first_support & second_support
    relative_depth = np.full(first_depth.shape, np.inf, dtype=np.float32)
    relative_depth[both] = (
        np.abs(first_depth[both] - second_depth[both])
        / np.minimum(first_depth[both], second_depth[both])
    )
    consistent = both & (relative_depth <= 0.02)
    conflict = both & ~consistent

    colour = np.zeros_like(first_colour)
    first_only = first_support & ~second_support
    second_only = second_support & ~first_support
    colour[first_only] = first_colour[first_only]
    colour[second_only] = second_colour[second_only]
    colour[consistent] = (
        first_colour[consistent].astype(np.uint16) + second_colour[consistent].astype(np.uint16)
    ) // 2
    nearer_first = conflict & (first_depth <= second_depth)
    nearer_second = conflict & ~nearer_first
    colour[nearer_first] = first_colour[nearer_first]
    colour[nearer_second] = second_colour[nearer_second]

    write_supported_png(output / f"{label}-supported.png", colour, either)
    write_supported_png(output / f"{label}-mutual.png", colour, consistent)
    shutil.copy2(images / first.name, output / f"{label}-nearest.jpg")
    total = camera.width * camera.height
    return {
        "label": label,
        "first": first.name,
        "second": second.name,
        "sharedSparseLandmarks": pair["shared"],
        "cameraDistance": pair["distance"],
        "cameraAngularDistanceDegrees": pair["angle"],
        "supportedByAtLeastOnePercent": float(either.sum() / total * 100),
        "supportedByTwoPercent": float(both.sum() / total * 100),
        "mutuallyConsistentPercent": float(consistent.sum() / total * 100),
        "occlusionOrConflictPercent": float(conflict.sum() / total * 100),
        "unsupportedPercent": float((~either).sum() / total * 100),
    }


def create_photo_graph(views: list[View], pairs: list[dict], images: Path, output: Path) -> list[dict]:
    graph_dir = output / "photos"
    graph_dir.mkdir(exist_ok=True)
    for view in views:
        image = cv2.imread(str(images / view.name))
        thumbnail = cv2.resize(image, (800, 600), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(graph_dir / view.name), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 88])

    graph = []
    for view in views:
        neighbours = []
        for pair in pairs:
            if pair["first"] is view:
                other = pair["second"]
            elif pair["second"] is view:
                other = pair["first"]
            else:
                continue
            neighbours.append({
                "name": other.name,
                "shared": pair["shared"],
                "distance": pair["distance"],
                "angle": pair["angle"],
                "score": pair["score"],
            })
        graph.append({
            "name": view.name,
            "center": view.center.tolist(),
            "forward": view.forward.tolist(),
            "neighbours": sorted(neighbours, key=lambda item: item["score"], reverse=True)[:4],
        })
    return graph


def create_html(
    output: Path,
    graph: list[dict],
    comparisons: list[dict],
    planar_proxy: dict | None,
) -> None:
    payload = json.dumps(json.dumps(graph))
    cards = []
    for comparison in comparisons:
        label = comparison["label"]
        cards.append(f"""
        <section>
          <h2>{html.escape(comparison["first"])} → {html.escape(comparison["second"])}</h2>
          <div class="comparison">
            <figure><img src="{label}-nearest.jpg"><figcaption>Nearest captured photograph</figcaption></figure>
            <figure class="checker"><img src="{label}-supported.png"><figcaption>Depth-assisted source-pixel reprojection</figcaption></figure>
            <figure class="checker"><img src="{label}-mutual.png"><figcaption>Two-view consistent pixels only</figcaption></figure>
          </div>
          <pre>{html.escape(json.dumps(comparison, indent=2))}</pre>
        </section>""")
    planar = ""
    if planar_proxy:
        planar = f"""
        <section>
          <h2>Bounded planar proxy</h2>
          <div class="comparison">
            <figure class="checker"><img src="planar-proxy.png"><figcaption>Actual source pixels projected through the strongest recovered plane</figcaption></figure>
          </div>
          <pre>{html.escape(json.dumps(planar_proxy, indent=2))}</pre>
        </section>"""
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Remember representation spike</title>
<style>
body {{ margin: 0; background: #111; color: #eee; font: 16px system-ui; }}
header, section {{ padding: 20px; }}
#photo {{ display: block; width: min(100%, 1000px); max-height: 72vh; object-fit: contain; background: #000; }}
button {{ margin: 8px 8px 0 0; padding: 10px; }}
.comparison {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
figure {{ margin: 0; }} figure img {{ width: 100%; }}
.checker {{ background: repeating-conic-gradient(#444 0 25%, #222 0 50%) 50% / 20px 20px; }}
pre {{ white-space: pre-wrap; }} @media(max-width: 900px) {{ .comparison {{ grid-template-columns: 1fr; }} }}
</style>
<header>
  <h1>Spatial photograph graph</h1>
  <p>Every displayed pixel is captured. Navigation follows recovered camera positions and overlap.</p>
  <img id="photo"><div id="caption"></div><div id="nav"></div>
</header>
{''.join(cards)}
{planar}
<script>
const graph = JSON.parse({payload});
let current = 0;
function show(index) {{
  current = index;
  const view = graph[index];
  photo.src = "photos/" + view.name;
  caption.textContent = `${{index + 1}}/${{graph.length}} — ${{view.name}}`;
  nav.replaceChildren(...view.neighbours.map(n => {{
    const button = document.createElement("button");
    button.textContent = `Move toward ${{n.name}} (${{n.shared}} shared landmarks)`;
    button.onclick = () => show(graph.findIndex(v => v.name === n.name));
    return button;
  }}));
}}
show(0);
</script>"""
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-text", type=Path, required=True)
    parser.add_argument("--images-text", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--depth-maps", type=Path, required=True)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--visibility", type=Path)
    parser.add_argument("--mvs-order", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparisons", type=int, default=3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    camera = parse_camera(args.camera_text)
    views = parse_views(args.images_text)
    pairs = rank_pairs(views)
    graph = create_photo_graph(views, pairs, args.images, args.output)
    comparisons = [
        create_midpoint(pair, camera, args.images, args.depth_maps, args.output, f"view-{index + 1}")
        for index, pair in enumerate(pairs[:args.comparisons])
    ]
    if bool(args.ply) != bool(args.visibility):
        parser.error("--ply and --visibility must be provided together.")
    planes, plane_models = detect_planes(args.ply, args.visibility, len(views)) if args.ply else ([], [])
    planar_proxy = None
    if plane_models:
        if not args.mvs_order:
            parser.error("--mvs-order is required when measuring planar geometry.")
        mvs_names = [
            line.strip()
            for line in args.mvs_order.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("__") and "," not in line
        ]
        planar_proxy = create_planar_proxy(
            plane_models[0],
            views,
            mvs_names,
            camera,
            args.images,
            args.output,
            pairs[0],
        )
    metrics = {
        "views": len(views),
        "photoGraph": graph,
        "midpointComparisons": comparisons,
        "planarGeometry": planes,
        "planarProxy": planar_proxy,
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    create_html(args.output, graph, comparisons, planar_proxy)
    print(f"Wrote local representation comparison to {args.output}")


if __name__ == "__main__":
    main()
