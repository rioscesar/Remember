# Radiance-field representation spike (3D Gaussian Splatting)

This desktop-only Milestone 0.4 experiment tests whether an evidence-trained radiance representation (3D Gaussian Splatting, via `gsplat`) produces a more photographic spatial memory than Milestone 0.3's atlas compositor, while preserving provenance and privacy. It does not change the Android application, and no generative AI or hole-filling was used at any step.

Private photographs, checkpoints, renders, and the training environment remain outside Git. Only the generalized training/rendering script and aggregate measurements are published.

## Method

1. Reuse the existing `sparse-strong` COLMAP reconstruction (registered camera poses + sparse point cloud) — no new structure-from-motion work was performed for this experiment.
2. Initialize one 3D Gaussian per recovered sparse point (position, per-point nearest-neighbor scale, identity-ish rotation, direct RGB color, opacity), with no synthesized or hallucinated Gaussians.
3. Train with `gsplat`'s `rasterization` + `DefaultStrategy` (standard 3DGS adaptive density control: gradient-driven splitting/duplication, opacity pruning, periodic opacity reset), optimizing plain per-pixel L1 photometric loss against the training photographs only.
4. Render four evidence categories per photograph/pose: (A) an original training camera pose, (B) one held-out registered photograph never used in training, (C) three interpolated poses between the two closest training cameras, (D) one pose deliberately extrapolated beyond the recovered camera trajectory.
5. For every render, also save the accumulated-alpha (coverage/confidence) map and record `meanAccumulatedAlpha`, `lowConfidencePixelPercent`, and `distanceToNearestTrainingCamera`.

No pixel is generated, inpainted, or extended: every rendered pixel is a weighted blend of Gaussians whose color parameters were fit exclusively to the registered source photographs.

## Dataset

The `sparse-strong` reconstruction already had 11 of 13 photos registered and 2,631 sparse points (single shared PINHOLE camera), a legitimate improvement made in a prior milestone — not new work for this experiment. One registered image was reserved as a holdout, leaving 10 training views.

## Training

| Item | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 3070 Laptop (8 GB) |
| Stack | torch 2.1.2+cu118, gsplat 1.5.2+pt21cu118 |
| Render resolution | 1000×750 (½ downsample) |
| Iterations | 7,000 |
| Wall-clock training time | 78.5 s |
| Initial Gaussians | 2,631 (from sparse points) |
| Final Gaussians | 197,133 |
| Final training loss (L1) | 0.016 (from 0.257 at step 0) |

Training was fast and numerically stable on this hardware; no crashes, NaNs, or runaway divergence occurred across 7,000 iterations.

## Results

| Render | PSNR vs. photo | Mean alpha | Low-confidence pixels | Distance to nearest training camera |
|---|---:|---:|---:|---:|
| A: training pose | 26.02 dB | 0.932 | 4.6% | 0.0 |
| B: holdout pose | 12.79 dB | 0.928 | 4.2% | 1.96 |
| C: interpolated 50% | — | 0.919 | 6.0% | 0.078 |
| D: extrapolated | — | 0.584 | 40.3% | 6.31 |

**A (training pose):** genuinely photographic and recognizable — the kitchen layout, cabinets, appliances, and floor are all identifiable, a clear improvement over Milestone 0.3's fragmented atlas. PSNR 26 dB confirms a strong direct fit.

**B (holdout pose) and C (interpolated):** both degrade into dense needle/spike-shaped Gaussian artifacts. The overall room layout is still faintly recognizable in C (closest to a training camera), but B is dominated by streaking artifacts and its PSNR (12.8 dB) is a near-failure. This is classic 3DGS overfitting: 10 training views is far below the number (typically 50–300+) needed for stable novel-view generalization.

**D (extrapolated):** total failure. The render is unrecognizable colored streaks with no photographic content. Critically, the alpha/confidence map does **not** cleanly fade to transparent in unsupported regions — it shows large, confidently opaque (bright) areas overlapping the garbage pixels. This means low geometric support is not reliably self-reporting as "unsupported": the representation can be confidently wrong, which is a meaningful failure against the founder brief's "unsupported areas must remain transparent/absent" requirement.

## Evidence artifacts

Each local run writes (private, not committed): `checkpoint.pt`, `metrics.json`, and RGB + alpha PNGs for all four render categories.

## Comparison vs. Milestone 0.3

| | Milestone 0.3 (atlas) | Milestone 0.4 (Gaussian splatting) |
|---|---|---|
| At/near training views | Fragmented, exposure seams, black holes | Photographic, recognizable, PSNR 26 dB |
| Away from training views | N/A (no novel-view synthesis attempted) | Severe artifacting; near-failure at only ~2 units away |
| Unsupported-area honesty | Explicitly transparent (no confident wrong pixels) | Not reliable — can be confidently wrong |
| Generated content | None | None |

Milestone 0.4 is a **quality improvement directly at captured viewpoints** but a **regression in honesty about its own uncertainty**, and it has not yet been shown to generalize to the free navigation Remember ultimately needs.

## Mobile/Android feasibility (discussion only, not implemented)

197K Gaussians rendered in well under real-time on a discrete GPU; on-device Android rendering was not evaluated and would need a mobile-appropriate rasterizer, compression, and likely 10–50x fewer Gaussians. No Android work was done or is proposed here.

## Licensing

`gsplat` (nerfstudio-project) is **Apache-2.0** and was used instead of the original INRIA `gaussian-splatting` reference implementation, which is licensed for non-commercial research/evaluation only and is unsuitable for Remember's proprietary, publicly-visible-source product.

## Verdict: **B — PARTIAL**

Radiance-field splatting produces materially better photographic quality than Milestone 0.3 at or near captured camera poses, but with only 10–11 registered photographs it does not generalize to novel viewpoints, and its confidence signal is not trustworthy for distinguishing supported from unsupported space. It is not ready to replace or extend the product representation. A follow-up would need either substantially more registered viewpoints per space, or an explicit per-pixel provenance/uncertainty gate layered on top of the raw alpha channel, before this technique could be considered for Remember's "evidence before imagination" guarantee.
