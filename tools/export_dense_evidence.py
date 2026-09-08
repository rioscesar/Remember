#!/usr/bin/env python3
"""Export only multi-view-supported COLMAP fused points for Remember."""

import argparse
import json
import math
import struct
from collections import Counter
from pathlib import Path

MIN_VIEWS = 3
MIN_POINTS = 1_000
MIN_VOLUME_RATIO = 0.05
# A point on a real surface sits among other points recovered from the same surface.
# Isolated points are stereo noise, which appears on textureless walls where matching
# is unreliable. This only ever removes points; it never moves or invents them.
NEIGHBOUR_RADIUS_FRACTION = 1 / 250
MIN_NEIGHBOURS = 4
# Depth is only well constrained when the supporting photographs see a point from
# genuinely different directions. Near-parallel viewing rays let a point slide along
# the ray, which appears as a streak of false geometry across textureless walls.
MIN_TRIANGULATION_DEGREES = 5.0


def parse_mvs_order(path: Path) -> list[str]:
    """Image names in the order COLMAP's visibility indices refer to."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("__") and "," not in line
    ]


def max_triangulation_angle(
    point: tuple[float, float, float],
    centers: list[tuple[float, float, float]],
) -> float:
    """Largest angle, in degrees, subtended at the point between two supporting cameras."""
    rays = []
    for center in centers:
        dx, dy, dz = center[0] - point[0], center[1] - point[1], center[2] - point[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length > 0:
            rays.append((dx / length, dy / length, dz / length))
    widest = 0.0
    for index, first in enumerate(rays):
        for second in rays[index + 1:]:
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second))))
            widest = max(widest, math.degrees(math.acos(dot)))
    return widest


def reject_isolated_points(points: list[dict], diagonal: float) -> list[dict]:
    """Drop points with too few neighbours to be part of a recovered surface."""
    radius = diagonal * NEIGHBOUR_RADIUS_FRACTION
    if radius <= 0:
        return points
    grid: dict[tuple[int, int, int], list[dict]] = {}
    for point in points:
        cell = (
            int(math.floor(point["x"] / radius)),
            int(math.floor(point["y"] / radius)),
            int(math.floor(point["z"] / radius)),
        )
        grid.setdefault(cell, []).append(point)

    kept = []
    squared = radius * radius
    for point in points:
        cx = int(math.floor(point["x"] / radius))
        cy = int(math.floor(point["y"] / radius))
        cz = int(math.floor(point["z"] / radius))
        neighbours = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if other is point:
                            continue
                        distance = (
                            (other["x"] - point["x"]) ** 2
                            + (other["y"] - point["y"]) ** 2
                            + (other["z"] - point["z"]) ** 2
                        )
                        if distance <= squared:
                            neighbours += 1
                            if neighbours >= MIN_NEIGHBOURS:
                                break
                    if neighbours >= MIN_NEIGHBOURS:
                        break
                if neighbours >= MIN_NEIGHBOURS:
                    break
            if neighbours >= MIN_NEIGHBOURS:
                break
        if neighbours >= MIN_NEIGHBOURS:
            kept.append(point)
    return kept


def read_exact(file, size: int, context: str) -> bytes:
    value = file.read(size)
    if len(value) != size:
        raise RuntimeError(f"Truncated {context}.")
    return value


def read_ply(path: Path) -> list[tuple[float, float, float, int, int, int]]:
    with path.open("rb") as file:
        lines = []
        while True:
            line = file.readline()
            if not line:
                raise RuntimeError("PLY header ended before end_header.")
            lines.append(line.decode("ascii").rstrip("\r\n"))
            if lines[-1] == "end_header":
                break
        count = int(next(line.split()[2] for line in lines if line.startswith("element vertex")))
        record = struct.Struct("<ffffffBBB")
        vertices = []
        for point_index in range(count):
            x, y, z, _, _, _, red, green, blue = record.unpack(
                read_exact(file, record.size, f"PLY vertex {point_index}")
            )
            if not all(math.isfinite(value) for value in (x, y, z)):
                raise RuntimeError(f"PLY vertex {point_index} has non-finite coordinates.")
            vertices.append((x, y, z, red, green, blue))
        if file.read(1):
            raise RuntimeError("PLY contains unexpected data after its declared vertices.")
        return vertices


def read_visibility(path: Path, max_views: int) -> list[set[int]]:
    with path.open("rb") as file:
        count, = struct.unpack("<Q", read_exact(file, 8, "visibility header"))
        visibility = []
        for point_index in range(count):
            view_count, = struct.unpack("<I", read_exact(file, 4, f"visibility count {point_index}"))
            if view_count > max_views:
                raise RuntimeError(f"Invalid visibility count {view_count} at point {point_index}.")
            views = struct.unpack(
                f"<{view_count}I",
                read_exact(file, view_count * 4, f"visibility indices {point_index}"),
            )
            visibility.append(set(views))
        if file.tell() != path.stat().st_size:
            raise RuntimeError("Visibility sidecar contains trailing or unparsed data.")
        return visibility


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, ...], ...]:
    """World-to-camera rotation from a COLMAP quaternion."""
    return (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )


def camera_center(fields: list[str]) -> tuple[float, float, float]:
    qw, qx, qy, qz, tx, ty, tz = (float(value) for value in fields[1:8])
    rotation = rotation_matrix(qw, qx, qy, qz)
    return tuple(-sum(rotation[row][column] * (tx, ty, tz)[row] for row in range(3)) for column in range(3))


def parse_registered_images(path: Path) -> tuple[list[str], list[tuple[float, float, float]], list[dict]]:
    names = []
    centers = []
    poses = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[0].isdigit() and fields[8].isdigit():
            name = " ".join(fields[9:])
            center = camera_center(fields)
            names.append(name)
            centers.append(center)
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            tx, ty, tz = (float(value) for value in fields[5:8])
            poses.append({
                "name": name,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "cx": center[0], "cy": center[1], "cz": center[2],
            })
    poses.sort(key=lambda pose: pose["name"])
    return sorted(names), centers, poses


def parse_focal_normalized(path: Path) -> float | None:
    """Focal length as a fraction of image width, so the viewer can match the photographs' field of view."""
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit():
            width = float(fields[2])
            focal = float(fields[4])
            if width > 0 and math.isfinite(focal):
                return focal / width
    return None


def median_camera_baseline(centers: list[tuple[float, float, float]]) -> float:
    distances = [
        math.dist(first, second)
        for index, first in enumerate(centers)
        for second in centers[index + 1:]
    ]
    return percentile(distances, 0.5)


def principal_spread(points: list[dict]) -> list[float]:
    """Standard deviation along each principal axis, largest first."""
    count = len(points)
    axes = ("x", "y", "z")
    mean = [sum(point[axis] for point in points) / count for axis in axes]
    covariance = [
        [
            sum((point[axes[row]] - mean[row]) * (point[axes[column]] - mean[column]) for point in points) / count
            for column in range(3)
        ]
        for row in range(3)
    ]
    for _ in range(100):
        row, column = max(
            ((i, j) for i in range(3) for j in range(3) if i < j),
            key=lambda pair: abs(covariance[pair[0]][pair[1]]),
        )
        if abs(covariance[row][column]) < 1e-12:
            break
        theta = (covariance[column][column] - covariance[row][row]) / (2 * covariance[row][column])
        tangent = (1 if theta >= 0 else -1) / (abs(theta) + math.sqrt(theta * theta + 1))
        cosine = 1 / math.sqrt(tangent * tangent + 1)
        sine = tangent * cosine
        for k in range(3):
            left = cosine * covariance[k][row] - sine * covariance[k][column]
            right = sine * covariance[k][row] + cosine * covariance[k][column]
            covariance[k][row], covariance[k][column] = left, right
        for k in range(3):
            left = cosine * covariance[row][k] - sine * covariance[column][k]
            right = sine * covariance[row][k] + cosine * covariance[column][k]
            covariance[row][k], covariance[column][k] = left, right
    return sorted((math.sqrt(max(covariance[i][i], 0.0)) for i in range(3)), reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export multi-view-supported COLMAP dense points.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True, help="Direct source raster directory")
    parser.add_argument("--registered-images-text", type=Path, required=True)
    parser.add_argument(
        "--mvs-order",
        type=Path,
        help="dense/stereo/patch-match.cfg, giving the image order the visibility indices use",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_names = sorted(path.name for path in args.source_images.iterdir() if path.is_file())
    registered_names, camera_centers, camera_poses = parse_registered_images(args.registered_images_text)
    if len(registered_names) < MIN_VIEWS:
        raise RuntimeError("The dense workspace has too few registered photographs.")
    vertices = read_ply(args.ply)
    visibility = read_visibility(args.visibility, len(registered_names))
    if len(vertices) != len(visibility):
        raise RuntimeError("COLMAP point and visibility record counts do not agree.")
    accepted = []
    weak_triangulation = 0
    mvs_names = parse_mvs_order(args.mvs_order) if args.mvs_order else []
    centers_by_name = {
        pose["name"]: (pose["cx"], pose["cy"], pose["cz"]) for pose in camera_poses
    }
    mvs_centers = [centers_by_name.get(name) for name in mvs_names]
    if mvs_names and (
        len(mvs_names) != len(registered_names) or any(center is None for center in mvs_centers)
    ):
        raise RuntimeError("MVS image order does not match the registered camera poses.")
    multi_view_points = 0
    for vertex, views in zip(vertices, visibility):
        if len(views) < MIN_VIEWS:
            continue
        multi_view_points += 1
        x, y, z, red, green, blue = vertex
        if mvs_centers:
            supporting = [
                mvs_centers[index]
                for index in views
                if 0 <= index < len(mvs_centers) and mvs_centers[index] is not None
            ]
            if len(supporting) >= 2 and max_triangulation_angle((x, y, z), supporting) < MIN_TRIANGULATION_DEGREES:
                weak_triangulation += 1
                continue
        accepted.append({
            "x": x, "y": y, "z": z, "r": red, "g": green, "b": blue,
            "support": len(views),
        })
    if len(accepted) < MIN_POINTS:
        raise RuntimeError("Too little multi-view-supported dense geometry was recovered.")
    spans = [
        percentile([point[axis] for point in accepted], 0.95) - percentile([point[axis] for point in accepted], 0.05)
        for axis in ("x", "y", "z")
    ]
    accepted = reject_isolated_points(accepted, math.sqrt(sum(span * span for span in spans)))
    if len(accepted) < MIN_POINTS:
        raise RuntimeError("Too little of the dense geometry forms connected surfaces.")
    spans = [
        percentile([point[axis] for point in accepted], 0.95) - percentile([point[axis] for point in accepted], 0.05)
        for axis in ("x", "y", "z")
    ]
    robust_diagonal = math.sqrt(sum(span * span for span in spans))
    median_baseline = median_camera_baseline(camera_centers)
    if median_baseline < 0.001:
        raise RuntimeError("Recovered cameras have no usable spatial baseline.")
    spread_ratio = robust_diagonal / median_baseline
    if spread_ratio < 0.5:
        raise RuntimeError("Dense geometry is too concentrated relative to camera baseline.")
    spread = principal_spread(accepted)
    volume_ratio = spread[2] / spread[0] if spread[0] > 0 else 0.0
    if volume_ratio < MIN_VOLUME_RATIO:
        raise RuntimeError(
            "Recovered geometry is a single flat surface rather than an explorable place "
            f"(thinnest principal spread is {volume_ratio:.1%} of the widest)."
        )
    output = {
        "formatVersion": 1,
        "sourceImages": source_names,
        "registeredPhotoNames": registered_names,
        "rejectedPhotoNames": sorted(set(source_names) - set(registered_names)),
        "cameraPoses": camera_poses,
        "focalLengthNormalized": parse_focal_normalized(
            args.registered_images_text.with_name("cameras.txt")
        ),
        "diagnostics": {
            "qualityGatePassed": True,
            "qualityGateFailures": [],
            "triangulatedLandmarks": len(vertices),
            "retainedLandmarks": len(accepted),
            "representation": "dense-geometric-fusion",
            "minimumDistinctViews": MIN_VIEWS,
            "candidatePoints": len(vertices),
            "multiViewPoints": multi_view_points,
            "weaklyTriangulatedRejected": weak_triangulation,
            "isolatedPointsRejected": multi_view_points - len(accepted),
            "acceptedPoints": len(accepted),
            "supportHistogram": dict(sorted(Counter(point["support"] for point in accepted).items())),
            "landmarkRobustSpans": spans,
            "landmarkRobustDiagonal": robust_diagonal,
            "medianCameraBaseline": median_baseline,
            "landmarkDiagonalToMedianBaseline": spread_ratio,
            "principalSpread": spread,
            "thinnestToWidestPrincipalSpread": volume_ratio,
        },
        "points": accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(accepted)} multi-view-supported dense points to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, struct.error) as error:
        raise SystemExit(f"Dense export stopped: {error}")
