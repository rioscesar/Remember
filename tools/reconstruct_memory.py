#!/usr/bin/env python3
"""Create an evidence-preserving sparse reconstruction bundle with local COLMAP."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

MIN_REGISTERED_IMAGES = 3
MIN_POINTS = 100
MIN_POINT_SUPPORT = 3


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def parse_images(path: Path) -> tuple[set[int], set[str]]:
    ids: set[int] = set()
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 10 and fields[0].isdigit() and fields[8].isdigit():
            ids.add(int(fields[0]))
            names.add(" ".join(fields[9:]))
    return ids, names


def parse_points(path: Path) -> tuple[int, list[dict]]:
    total = 0
    points: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        total += 1
        observations = fields[8:]
        image_ids = {int(observations[index]) for index in range(0, len(observations), 2)}
        error = float(fields[7])
        if len(image_ids) < MIN_POINT_SUPPORT or error < 0:
            continue
        points.append(
            {
                "x": float(fields[1]),
                "y": float(fields[2]),
                "z": float(fields[3]),
                "r": int(fields[4]),
                "g": int(fields[5]),
                "b": int(fields[6]),
                "support": len(image_ids),
                "reprojectionError": error,
            }
        )
    return total, points


def largest_model(sparse_dir: Path) -> Path:
    models = [path for path in sparse_dir.iterdir() if path.is_dir()]
    if not models:
        raise RuntimeError("COLMAP could not register a reconstructable group of photographs.")
    return max(models, key=lambda path: (path / "points3D.bin").stat().st_size if (path / "points3D.bin").exists() else 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local CPU COLMAP sparse reconstruction and export Remember evidence JSON."
    )
    parser.add_argument("--images", type=Path, required=True, help="Directory containing one place's photos")
    parser.add_argument("--output", type=Path, required=True, help="Destination reconstruction.json")
    parser.add_argument("--colmap", default="colmap", help="Path to a locally installed COLMAP executable")
    parser.add_argument("--work", type=Path, help="Local temporary work directory (deleted after success; retained on failure)")
    args = parser.parse_args()

    images = args.images.resolve()
    if not images.is_dir():
        raise RuntimeError(f"Image directory does not exist: {images}")
    source_images = sorted(
        path.name for path in images.iterdir()
        if path.is_file() and path.suffix.lower() in {".avif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    )
    if len(source_images) < MIN_REGISTERED_IMAGES:
        raise RuntimeError("Choose at least three photos of one place.")

    work = args.work.resolve() if args.work else args.output.resolve().parent / ".remember-work"
    if work.exists():
        raise RuntimeError(f"Work directory already exists: {work}. Choose an empty local directory.")
    database = work / "database.db"
    sparse = work / "sparse"
    text = work / "sparse-text"
    image_list = work / "source-images.txt"
    work.mkdir(parents=True)
    succeeded = False
    try:
        sparse.mkdir()
        text.mkdir()
        image_list.write_text("\n".join(source_images) + "\n", encoding="utf-8")
        run([
            args.colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(images),
            "--image_list_path", str(image_list),
            "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.use_gpu", "0",
            "--FeatureExtraction.max_image_size", "1600", "--FeatureExtraction.num_threads", "4",
            "--SiftExtraction.max_num_features", "4096",
        ])
        run([args.colmap, "exhaustive_matcher", "--database_path", str(database), "--FeatureMatching.use_gpu", "0"])
        run([
            args.colmap, "mapper", "--database_path", str(database), "--image_path", str(images),
            "--output_path", str(sparse), "--Mapper.min_model_size", str(MIN_REGISTERED_IMAGES),
            "--Mapper.ba_use_gpu", "0",
        ])
        model = largest_model(sparse)
        run([
            args.colmap, "model_converter", "--input_path", str(model), "--output_path", str(text),
            "--output_type", "TXT",
        ])
        registered_ids, registered_names = parse_images(text / "images.txt")
        total_points, points = parse_points(text / "points3D.txt")
        if len(registered_ids) < MIN_REGISTERED_IMAGES:
            raise RuntimeError("Not enough photographs could be registered together. Try photos with more overlap.")
        if len(points) < MIN_POINTS:
            raise RuntimeError("Not enough shared spatial evidence was recovered. Remember will not create a scene.")
        payload = {
            "formatVersion": 1,
            "sourceImages": source_images,
            "registeredPhotoNames": sorted(registered_names),
            "rejectedPhotoNames": sorted(set(source_images) - registered_names),
            "diagnostics": {
                "triangulatedLandmarks": total_points,
                "retainedLandmarks": len(points),
            },
            "points": points,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"Wrote {len(points)} of {total_points} triangulated landmarks with sufficient evidence to {args.output}")
        succeeded = True
        return 0
    finally:
        if succeeded and work.exists():
            shutil.rmtree(work)
        elif work.exists():
            print(f"Retained diagnostic workspace: {work}", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Reconstruction stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
