# Evidence-preserving photographic coherence

This desktop-only Milestone 0.3 experiment tests whether the accepted semantic wall can be made to read as one coherent photograph while preserving direct source traceability. It does not change the Android application.

Private photographs, masks, atlases, contributor weights, renders, videos, reconstruction artifacts, and identifying metadata remain outside Git. Only the generalized implementation and aggregate measurements are published.

## Method

1. Reuse the strongest recovered wall plane and the semantic/geometry gates from the accepted semantic-plane experiment.
2. Project each accepted photograph into one canonical wall-coordinate atlas. This makes source selection independent of the virtual camera and prevents temporal source flicker.
3. Apply bounded per-channel gain normalization. No spatially varying correction or synthesized texture is used.
4. Rank source observations using plane incidence, projected sampling density, homography inliers and RMS, camera distance, and distance from rejected-mask boundaries.
5. Give the highest-ranked valid observation 75% of a texel's weight. Photometrically compatible observations, within 25 mean RGB levels of the primary observation, may share the remaining 25%. Incompatible observations are rejected.
6. Preserve a primary-owner raster, contributor-count raster, and complete per-source weight stack for the private output.
7. Render a 31-frame sweep between the closest accepted recovered cameras. Unsupported texels stay transparent.

The experiment uses only captured source pixels and deterministic image transformations. It does not inpaint, extend textures, infer missing appearance, synthesize pixels, or introduce generated geometry.

## Results

Seven source photographs passed the existing wall evidence gate; two were rejected. The sweep spans 0.164 reconstruction units and 8.2 degrees, so it tests deliberately small local motion rather than unrestricted navigation.

| Measurement | Result |
|---|---:|
| Recovered atlas covered by source photographs | 79.86% |
| Raw multi-view atlas support | 70.29% |
| Raw single-view atlas support | 9.57% |
| Atlas pixels with multiple compatible contributors | 67.53% |
| Pre-normalization overlap RGB disagreement | 18.59 |
| Post-normalization overlap RGB disagreement | 14.45 |
| Before-normalization ownership-boundary RGB gradient | 11.77 |
| Final ownership-boundary RGB gradient | 9.70 |
| Supported atlas area containing a rejected source conflict | 30.52% |
| Full-frame direct source pixels across sweep | 8.87–10.36% |
| Full-frame compatible multi-source pixels across sweep | 7.75–9.12% |
| Full-frame single-source pixels across sweep | 1.08–1.24% |
| Canonical ownership switches across 31 frames | 0 |
| Generated pixels | 0 |

Bounded normalization reduced mean overlap disagreement by 22.3%, and the final compositor reduced the measured ownership-boundary gradient by 17.6%. Canonical ownership remained unchanged for every frame.

## Evidence artifacts

Each local run writes:

- unnormalized and normalized wall atlases plus a side-by-side comparison;
- primary-owner and colorized ownership rasters;
- a contributor-count raster and compressed per-source weight stack;
- conflict and provenance visualizations;
- 31 transparent RGBA sweep frames and a local HTML animation that displays transparency over a checkerboard;
- aggregate `metrics.json`.

These outputs derive from private photographs and must not be committed.

## Verdict

The experiment improves numerical photometric consistency and eliminates camera-dependent ownership flicker, but the wall still does not read as one continuous photograph. Large low-frequency exposure regions remain visible, semantic exclusions leave conspicuous black holes through wall art and foreground boundaries, and the top portion contains fragmented source transitions. Compatible blending softens some transitions but does not resolve those structural artifacts.

**Milestone 0.3 fails its perceptual success criterion.** The result is not evidence that the complete apartment can be represented coherently, and it is not ready for Android integration. The canonical atlas and contributor records are useful implementation findings, but further work requires a new founder-approved experiment focused on wall-attached-object representation, mask resolution, or globally consistent photometric calibration.

Per the founder stop condition, this experiment does not proceed to floor, ceiling, additional geometry, AR, generated content, or product representation changes.

## Reproduction

Install the pinned desktop dependencies:

```powershell
python -m pip install -r tools\requirements-representation.txt
```

Run `tools/wall_coherence_spike.py` with the local COLMAP text model, undistorted source images, depth maps, fused PLY, visibility sidecar, and `patch-match.cfg`. Always choose an output directory outside the repository.
