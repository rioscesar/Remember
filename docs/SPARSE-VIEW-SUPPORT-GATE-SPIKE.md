# Sparse-view regularization and independent support gating

This desktop-only Milestone 0.5 experiment asks whether an evidence-compatible sparse-view regularization can make radiance-field reconstruction (Milestone 0.4) more robust for Remember's legacy-photo-count scenes, and whether a support/confidence signal *independent of Gaussian opacity* can honestly identify unsupported space. It does not change the Android application. No monocular depth network, no diffusion/generative model, and no CUDA-toolkit compile step were used.

Private photographs, checkpoints, and renders remain outside Git. Only the generalized scripts and aggregate measurements are published.

## Phase 1: sparse-view method research

| Candidate | Category | Verdict |
|---|---|---|
| SparseGS | Inference-assisted | Requires a monocular depth model (INFERRED geometry) and compiling the INRIA `diff-gaussian-rasterization` CUDA extension, which is non-commercial-licensed and needs a CUDA toolkit/MSVC we don't have locally. Rejected for this milestone. |
| DNGaussian | Inference-assisted | Same monocular-depth (DPT) dependency and the same INRIA-licensed/CUDA-compile requirement as SparseGS. Rejected. |
| FewViewGS, SE-GS | Inference-assisted / ensembling | Rely on pseudo-view synthesis or learned priors beyond plain multi-view geometry; not evaluated further given the licensing/compile blockers already found in the two closest published baselines. |
| **Evidence-compatible custom regularization (selected)** | Evidence-compatible | Uses only signals already in the COLMAP reconstruction: triangulated sparse-point depth (multi-view geometry, not learned), Gaussian scale/opacity priors, and an early-stopped adaptive-density schedule. No new heavy dependency, reuses the existing Apache-2.0 `gsplat` stack. |

Every published sparse-view framework surveyed depends on a monocular depth network and/or the INRIA non-commercial CUDA kernel. Both are incompatible with this milestone's evidence-compatible/licensing constraints and local compile environment (no `nvcc`/MSVC). This is itself a useful finding: a credible "off-the-shelf" sparse-view upgrade is not currently available without accepting an INFERRED geometry source or an incompatible license, so Milestone 0.5 built a smaller custom regularizer instead.

## Phase 2: the comparison

Question: **does multi-view-geometry regularization (no monocular priors) materially improve novel-view stability over vanilla 3DGS on the same Remember dataset?**

Two variants were trained back-to-back on the identical dataset/split:

- **Vanilla** — Milestone 0.4's unmodified training loop.
- **Regularized** — adds (a) a sparse-depth consistency loss supervising rendered expected-depth against each training photo's own triangulated COLMAP points (pure multi-view geometry, no monocular network), (b) a Gaussian scale-cap penalty targeting the needle/floater shapes seen in Milestone 0.4, (c) an opacity-entropy penalty discouraging semi-transparent haze, and (d) an earlier adaptive-density stop (`refine_stop_iter` at 40% of training) plus a tighter `prune_scale3d`.

## Phase 3: dataset

Unchanged from Milestone 0.4: the `sparse-strong` reconstruction, 11/13 registered images, 2,631 sparse points, same 10-view/1-holdout split, same render resolution.

## Phase 4: independent support gate

Built a per-pixel support classifier that **never reads Gaussian opacity**. For every rendered pixel: the rendered expected depth (`gsplat` `RGB+ED` mode) is unprojected to a 3D world point, then scored purely from camera geometry — distance and viewing-angle from that point to the *nearest training camera* — into `SUPPORTED` (≥0.55), `WEAKLY_SUPPORTED` (0.2–0.55), or `UNSUPPORTED` (<0.2). A smooth fade (not a hard cutoff) attenuates alpha between the weak/unsupported thresholds.

## Phase 5: honest fade prototype

Applied the support gate to both variants' holdout and extrapolated renders, producing raw vs. support-gated RGB/alpha pairs.

## Phase 6: results

| Metric | Vanilla | Regularized |
|---|---:|---:|
| Training time | 73.5 s | 96.4 s |
| Final Gaussian count | 193,858 (still growing at cutoff) | 79,062 (growth halted at iter ~2,800, stable afterward) |
| Training-pose PSNR / SSIM | 26.36 dB / 0.815 | 23.16 dB / 0.775 |
| Holdout-pose PSNR / SSIM | 12.21 dB / 0.407 | 12.61 dB / 0.475 |
| Extrapolated: raw mean alpha in gate-labeled unsupported region | 0.311 | 0.637 |
| Extrapolated: gated mean alpha in that same region | 0.0 | 0.0 |
| Extrapolated: % pixels gate-classified unsupported | 65.9% | 91.0% |

**Regularization achieved its primary engineering goal** (stopping runaway Gaussian growth, from 194K unstable/still-growing to 79K stable) and gave a modest, real SSIM improvement at the holdout pose (0.41 → 0.47), but **did not fix the underlying visual failure**: the holdout and interpolated renders for both variants remain dominated by needle/streak artifacts and are not recognizable as the apartment. The regularized variant also introduced a new, isolated blob-shaped floater artifact visible even at a *training* pose that the vanilla variant did not show — an example of one regularizer trading one artifact class for another rather than eliminating artifacts outright.

**The support gate is the clear success of this milestone.** Regardless of how confident the raw renderer was (alpha 0.31–0.64 in garbage regions), the independent gate correctly drove those regions to fully transparent (mean gated alpha 0.0) in both variants, and did so more aggressively/correctly for the more-stable regularized model (91% of the extrapolated view correctly flagged unsupported, vs. 66% for vanilla). This directly answers Milestone 0.4's core complaint: **splat opacity is not trustworthy, but a camera-geometry-only gate can be.**

For the holdout pose, most pixels fall into `WEAKLY_SUPPORTED` (88–92%) rather than cleanly `SUPPORTED` or `UNSUPPORTED`, so the gate only partially fades that view rather than hiding it — this is an honest reflection of the fact that the holdout camera is genuinely close to (not far from) the training cameras, so a geometry-only gate correctly does not call it "unsupported" even though the *rendered content* is bad. This is a real limitation: the support gate detects "am I far from evidence," not "is the reconstruction visually correct," and those two questions are not the same. A future milestone should treat this as a distinct, still-open problem.

## Provenance

All rendered pixels remain RECONSTRUCTED (fit only from registered photographs and their own multi-view triangulated depths). The support classification uses only camera geometry — no INFERRED (monocular) or IMAGINED (generative) signal was used anywhere in this milestone.

## Licensing

Same as Milestone 0.4: `gsplat` (Apache-2.0). No INRIA-licensed code, no new model weights, downloaded or otherwise.

## Compute

Both variants trained on the same RTX 3070 laptop GPU used in Milestone 0.4. Neither hit a VRAM ceiling nor became impractically slow (73–96 s each); migration to the RTX 3080 desktop was not required and is not requested.

## Verdict: **B — PARTIAL**

Sparse-view regularization measurably stabilizes training (no runaway Gaussian growth) and gives a real but small quality gain at the holdout pose, but does not make novel-view reconstruction visually usable from only ~10 photographs — this half of the milestone's two-part success criterion is not met. The independent support/confidence gate, however, works exactly as intended: it reliably suppresses confidently-wrong radiance output in genuinely unsupported space without ever consulting renderer opacity, which is a durable finding regardless of which radiance method Remember eventually uses. Neither half alone is sufficient per the founder's stated criterion, so the overall milestone remains **B-PARTIAL**, carrying forward a validated support-gating mechanism as reusable infrastructure for future radiance-representation work.

## Founder review artifacts

Renders for both variants (training/holdout/interpolated/extrapolated, raw and support-gated, plus the support-score visualization) are available locally and were shared for review; see the conversation for the exact private paths. None are committed to Git.
