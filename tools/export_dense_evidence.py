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


def camera_center(fields: list[str]) -> tuple[float, float, float]:
    qw, qx, qy, qz, tx, ty, tz = (float(value) for value in fields[1:8])
    rotation = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
        (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
        (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
    )
    return tuple(-sum(rotation[row][column] * (tx, ty, tz)[row] for row in range(3)) for column in range(3))


def parse_registered_images(path: Path) -> tuple[list[str], list[tuple[float, float, float]]]:
    names = []
    centers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 10 and fields[0].isdigit() and fields[8].isdigit():
            names.append(" ".join(fields[9:]))
            centers.append(camera_center(fields))
    return sorted(names), centers


def median_camera_baseline(centers: list[tuple[float, float, float]]) -> float:
    distances = [
        math.dist(first, second)
        for index, first in enumerate(centers)
        for second in centers[index + 1:]
    ]
    return percentile(distances, 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export multi-view-supported COLMAP dense points.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--visibility", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True, help="Direct source raster directory")
    parser.add_argument("--registered-images-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_names = sorted(path.name for path in args.source_images.iterdir() if path.is_file())
    registered_names, camera_centers = parse_registered_images(args.registered_images_text)
    if len(registered_names) < MIN_VIEWS:
        raise RuntimeError("The dense workspace has too few registered photographs.")
    vertices = read_ply(args.ply)
    visibility = read_visibility(args.visibility, len(registered_names))
    if len(vertices) != len(visibility):
        raise RuntimeError("COLMAP point and visibility record counts do not agree.")
    accepted = []
    for vertex, views in zip(vertices, visibility):
        if len(views) >= MIN_VIEWS:
            x, y, z, red, green, blue = vertex
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
    robust_diagonal = math.sqrt(sum(span * span for span in spans))
    median_baseline = median_camera_baseline(camera_centers)
    if median_baseline < 0.001:
        raise RuntimeError("Recovered cameras have no usable spatial baseline.")
    spread_ratio = robust_diagonal / median_baseline
    if spread_ratio < 0.5:
        raise RuntimeError("Dense geometry is too concentrated relative to camera baseline.")
    output = {
        "formatVersion": 1,
        "sourceImages": source_names,
        "registeredPhotoNames": registered_names,
        "rejectedPhotoNames": sorted(set(source_names) - set(registered_names)),
        "diagnostics": {
            "qualityGatePassed": True,
            "qualityGateFailures": [],
            "triangulatedLandmarks": len(vertices),
            "retainedLandmarks": len(accepted),
            "representation": "dense-geometric-fusion",
            "minimumDistinctViews": MIN_VIEWS,
            "candidatePoints": len(vertices),
            "acceptedPoints": len(accepted),
            "supportHistogram": dict(sorted(Counter(point["support"] for point in accepted).items())),
            "landmarkRobustSpans": spans,
            "landmarkRobustDiagonal": robust_diagonal,
            "medianCameraBaseline": median_baseline,
            "landmarkDiagonalToMedianBaseline": spread_ratio,
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
