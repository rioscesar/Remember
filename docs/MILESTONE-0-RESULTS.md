# Milestone 0 results

## Implemented path

`Create Memory → select local photos → run local Companion → import reconstruction → Enter → navigate → leave → reopen from library`

The Android app persists memory metadata and accepted sparse reconstruction data across restart. It provides an explicit import failure for malformed bundles, mismatched photos, too few source images, too few points, or unsupported landmarks. Companion reconstruction failure is explicit for insufficient overlap and does not produce a substitute scene.

## Genuine reconstruction

The representation is a sparse colored point cloud. Every rendered point originates from a COLMAP triangulated landmark with three or more observation tracks. The viewer uses the actual imported XYZ coordinates; rotation creates geometric parallax. No mesh, panorama, depth prediction, inferred texture, generative completion, or photo animation is used.

## Real-world validation — apartment photos (2026-09-07)

### Outcome: successful sparse reconstruction

The local CPU-only COLMAP 4.2.0 companion completed an evidence-backed sparse reconstruction from supplied ordinary apartment photographs. The clean validation run used only the 42 original HEIC files directly in the supplied source folder; it excluded eight nested JPEG duplicate exports. It generated a private `reconstruction.json` outside the repository and passed Remember's import contract.

| Measurement | Result |
|---|---:|
| Source photos | 42 |
| Registered photos | 8 (19.0%) |
| Rejected/unregistered photos | 34 |
| Triangulated landmarks before support filtering | 492 |
| Retained landmarks | 408 (82.9%) |
| Retained support distribution | 286 with 3 views; 108 with 4; 14 with 5 |
| Processing time | 67.47 seconds |
| Mean reprojection error | 0.939 pixels (maximum 2.658) |

The exported bundle is format version 1, has at least three registered source photos, contains only landmarks observed in at least three distinct photos, and exceeds the 100-landmark import threshold. The Android importer will additionally require that at least three selected source photos contributed to the reconstruction.

### Diagnostics and limitations

Feature extraction completed for all 42 original files. COLMAP found one usable eight-photo cluster and discarded several candidate reconstructions due to no or bad initial pairs. The 34 unregistered photos indicate limited overlap between the selected viewpoints, scene changes, low texture, repeated structure, viewpoint changes, or combinations of those factors. This validation did not inspect or preserve EXIF data, thumbnails, source imagery, or private point coordinates in Git.

The first attempt revealed an implementation bug: COLMAP recursively discovered eight nested JPEG duplicate exports even though the script inventoried only direct-folder originals. That attempt was discarded. The exporter now supplies COLMAP an explicit direct-file image list and was rerun successfully on exactly the 42 originals. This is a general correctness fix, not a dataset-specific threshold adjustment.

COLMAP core is BSD-licensed, but its license explicitly excludes its dependencies from that statement. The official `colmap-x64-windows-nocuda` 4.2.0 binary was used locally for validation only after its published SHA-256 was verified. It is not bundled or redistributed by Remember; distribution still requires a complete transitive dependency/license audit.

### Experiential validation — pending physical-device review

- **Recognition:** Can I tell what place I'm looking at without being told?
- **Orientation:** Can I understand where major objects/areas were relative to each other?
- **Exploration:** Does movement reveal meaningful spatial information?
- **Memory:** Does exploring it trigger recognition or recollection that simply viewing the source photos does not?
- **Experience:** Does it feel like the beginning of entering a memory, or merely like viewing a point cloud?

## Not yet demonstrated

This repository does not include sensitive sample imagery or a bundled COLMAP binary, so a physical-device end-to-end run and latency benchmark remain required before declaring Milestone 0 complete. AR portal work is intentionally deferred.
