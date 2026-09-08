# Reconstruction spike

**Decision:** Milestone 0 uses a private, offline desktop CPU COLMAP companion to reconstruct a sparse point cloud, then imports a provenance-preserving JSON bundle into Android. Full robust Structure-from-Motion directly on typical Android hardware is not credible for this milestone. The Android app is installable and usable locally, but processing requires the local companion.

## Candidates

| Pipeline | Device feasibility | License/dependencies | Output and risks | Decision |
|---|---|---|---|---|
| OpenCV feature matching plus custom SfM | Possible with NDK/JNI, but a robust unordered-gallery pipeline requires intrinsics, tracks, incremental mapping, bundle adjustment, and outlier management | Apache-2.0; native Android packaging; OpenCV SfM needs Ceres and has no Java wrapper | Sparse points; substantial engineering and correctness risk | Future spike |
| COLMAP 4.2 CPU sparse mapping | Practical on a desktop; not a drop-in Android AAR | BSD-style core but transitive dependencies require audit; headless native stack includes Ceres, Eigen, Boost, SQLite, image IO | Poses, colored landmarks, error, and observation tracks; 8–20 photos typically minutes on modern CPU | **Selected** |
| AliceVision/Meshroom | Desktop-only; dense processing favors NVIDIA GPU | MPL-2.0 plus dependencies | Textured mesh; high installation/hardware cost and evidence filtering still needed | Not tonight |
| OpenMVG/OpenMVS | Desktop native stack | MPL-2.0 plus OpenMVS AGPL and research-only IBFS concern | Sparse/dense results; unacceptable licensing/integration work | Rejected |
| Gaussian splatting | Requires prior poses/SfM and GPU training | Mixed licenses; common reference is non-commercial | Rich novel views can hide unsupported geometry | Future research |
| Monocular depth/novel view | Mobile inference is possible | Model licenses vary | Prediction is not multi-photo recovered geometry | Rejected |

## Selected pipeline

The companion uses COLMAP's CPU feature extraction, exhaustive matching, incremental mapper, and text conversion. It retains only landmarks observed by at least three images, with non-negative reprojection error. It requires at least three registered photos and 100 retained landmarks. These are initial product thresholds, not claims of scientific certainty.

`tools/reconstruct_memory.py` is the local exporter. It invokes only a local `colmap` executable, passes it an explicit list of supported image files directly in the selected folder (never nested folders), creates a temporary workspace beside the selected output, deletes that workspace after success, retains it after failure for local diagnostics, and writes `reconstruction.json` only after all thresholds pass.

Before export, the Companion additionally measures the recovered camera-center baselines and the robust (5th-to-95th percentile) landmark span. It rejects a result when fewer than half the selected photographs join its accepted component, camera centers have no measurable baseline, or the landmark diagonal is less than half the median camera baseline. These scale-relative checks prevent a small or collapsed fragment from being presented as an explorable place.

Expected CPU time is seconds to several minutes for 8–20 images resized to 1600 pixels, depending on texture, overlap, and hardware. It requires RAM appropriate to feature matching; a discrete GPU is not required for the selected sparse pipeline. It emits no metric distance because SfM translation is scale-ambiguous.

## Bundle and rendering

`reconstruction.json` carries version, selected source names, registered/rejected names, RGB XYZ points, support count, and reprojection error. Android verifies those limits and renders actual imported 3D landmarks with touch navigation. It fades points based on independent support and reprojection quality; it does not enlarge points into fake surfaces. Empty space remains empty.

## Dense multi-view evidence spike

The dense experiment uses a separate, local COLMAP 4.2.0 CUDA pipeline only after the Clean capture's sparse poses passed the quality gate. It undistorts the recovered component to 1600 pixels, runs PatchMatch with geometric consistency, and uses conservative geometric stereo fusion. No mesh reconstruction, hole filling, texture generation, depth completion, or generative process is allowed.

The fusion output is a candidate point cloud, not automatically accepted scene evidence. `tools/export_dense_evidence.py` reads the binary PLY and COLMAP visibility sidecar with bounded record reads, verifies that the sidecar is fully consumed, and retains a point only when at least three distinct MVS source views support it. It rejects truncated or malformed records, non-finite coordinates, too few registered cameras, insufficient retained points, absent camera baseline, and geometry whose robust spread is too concentrated relative to that baseline. Dense points preserve COLMAP's original coordinates and observed RGB values; they carry no synthetic reprojection error.

The CUDA executable remains a local validation dependency and is neither checked into nor redistributed by Remember. As with the CPU companion, production distribution requires a complete audit of COLMAP's transitive binaries and licenses.

## Privacy and risks

The pipeline is entirely local when COLMAP and Python are installed locally. The Android manifest has no network permission. Risks include poor overlap, repeated/textureless surfaces, moving subjects, mixed intrinsics, CPU time, unmeasured rendering performance, and licensing of a redistributed companion. The project ships the script only, not a COLMAP binary; a production redistribution requires a full transitive license audit.

## References

- [COLMAP CLI](https://colmap.github.io/cli.html)
- [COLMAP output format](https://colmap.github.io/format.html)
- [COLMAP tutorial: image overlap](https://colmap.github.io/tutorial.html)
- [COLMAP license](https://github.com/colmap/colmap/blob/4.2.0/COPYING.txt)
- [Android OpenGL ES](https://developer.android.com/develop/ui/views/graphics/opengl/about-opengl)
- [Android Auto Backup](https://developer.android.com/identity/data/autobackup)
