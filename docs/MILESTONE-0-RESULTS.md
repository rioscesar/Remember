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

### Physical-device result — not acceptable (2026-09-07)

On a Samsung Fold 6, the imported sparse scene appeared as a small, fragmented cluster on an otherwise empty dark canvas. It was not recognizable as the apartment and did not support meaningful orientation or exploration. This is a **product-validation failure**, not a successful experience merely because the bundle passed the import contract.

The geometry diagnosis agrees with the on-device result. The retained JPEG-derived cloud is concentrated in a narrow region: its central 90% spans approximately 0.01 reconstruction units on X, 0.01 on Y, and 0.04 on Z. A single Z outlier expands the full Z range to approximately 0.22 units. This is an insufficiently broad and spatially degenerate recovered cluster, rather than a representation of the apartment.

The viewer must not compensate by enlarging points into surfaces, inventing missing geometry, adding a panorama, or synthesizing texture. Reframing alone cannot turn this narrow cluster into an explorable memory.

**Smallest next engineering experiment:** preserve the same local COLMAP sparse pipeline and evidence contract, but add a companion-only reconstruction inspection report that measures component extent, camera baseline, and landmark distribution before export. Reject or label an import as insufficient when a component is too spatially degenerate to explore, then test a deliberately overlapping subset captured across one continuous apartment area. If that still produces only a narrow cluster, the next separate spike is evidence-validated dense multi-view stereo—not generated completion.

## Reconstruction quality gate and controlled-overlap diagnosis (2026-09-07)

The Companion now calculates and emits pre-export quality diagnostics from recovered camera centers and retained landmarks. It will not create an importable bundle unless at least half the selected photos join the accepted component, the median camera baseline is at least 0.001 reconstruction units, and the robust landmark diagonal is at least half that baseline. Android independently rejects any bundle whose quality gate did not pass.

### Full apartment set

The 42-photo JPEG validation set was reprocessed with the new gate. Its largest stable component registered 5 photos (11.9%), triangulated 286 landmarks, and retained 260 landmarks under the unchanged three-observation rule. The median recovered-camera baseline was 3.238 units, while the robust landmark diagonal was 1.294 units (ratio 0.400). The Companion correctly rejected it for both inadequate selected-photo coverage and concentrated geometry; it wrote no importable bundle.

### Controlled-overlap subset

An eight-photo subset was formed only from the images that had previously appeared in the earlier mixed-set component. This is a controlled reprocessing subset, not a newly captured photo set. COLMAP extracted features from all eight and found one connected match graph, but every attempted initialization was rejected as a bad initial pair; it created no sparse model.

This points primarily to an **input camera-geometry limitation**: the existing photographs do not provide a stable translated-view initial pair for the selected area. It also demonstrates expected SfM sensitivity to pair selection; the earlier mixed-set fragment must not be interpreted as a reliable reconstruction. The Android renderer is not the cause of this failure.

### Next controlled capture

To distinguish a capture limitation from a remaining SfM-pipeline limitation, make one new, continuous capture of a single apartment area: 12–20 still photos, walking a shallow arc with visible translation between shots, 60–80% overlap, and fixed exposure where possible. Include textured stationary features at multiple depths; avoid people, mirrors, blank walls, and large viewpoint jumps. This needs a new user capture because no supplied subset passes initial-pair geometry. Process that capture unchanged through the same local Companion and quality gate.

## Not yet demonstrated

This repository does not include sensitive sample imagery or a bundled COLMAP binary, so a physical-device end-to-end run and latency benchmark remain required before declaring Milestone 0 complete. AR portal work is intentionally deferred.
