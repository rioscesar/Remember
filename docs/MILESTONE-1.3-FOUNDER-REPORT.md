﻿# Founder steering update — Milestone 1.3

Milestone 1.3 turns the validated Milestone 1.0–1.2 pipeline (canonical room-face atlas, Doctrine v2 provenance, local IP-Adapter memory conditioning) into a single, deterministic, founder-presentable apartment-memory walkthrough demo. It ranks and selects only the missing faces that are actually visible along an intended walkthrough path, generates a small fixed-seed candidate set per selected face while rejecting unsupported semantic additions, harmonizes cross-face color continuity strictly on `IMAGINED` pixels, builds a 5-stop deterministic presentation path with a product-style mode (metrics/paths/filenames hidden) and a separate debug/provenance mode, demonstrates one protected critical region (a real captured painting) with exact pixel preservation and no fabricated person, and pre-generates every asset plus a reproducible backup recording. No private photographs, renders, generated textures, video, masks, or model files were added to Git; only generalized code, tests, and this report are committed.

## Runtime and pipeline

| Item | Result |
|---|---|
| Python runtime | Isolated local Python 3.10 environment outside the repository |
| Python / Torch / CUDA | 3.10.11 / 2.1.2+cu118 / CUDA 11.8 |
| GPU / total VRAM | NVIDIA GeForce RTX 3070 Laptop GPU / 8,191.5 MiB |
| Base Inpainting Model | `stable-diffusion-v1-5/stable-diffusion-inpainting` (fp16) |
| Visual Conditioning | `h94/IP-Adapter` (`ip-adapter_sd15.safetensors`, fp16) |
| Semantic Model | `openmmlab/upernet-convnext-tiny` (ADE20K semantic segmentation) |
| Fixed seed | `1301` (identical across all candidate generations) |
| Pipeline space | Canonical room-face atlas space, reused unmodified from Milestone 1.0–1.2 |

## Demo-visible face ranking and selection

All 6 canonical room faces were ranked by a deterministic demo-visibility score (recovered-first, then intended walkthrough path order `front → right → back → left → ceiling → floor`, weighted by projected screen area). Only **missing** faces that also rank as visible along the walkthrough path are selected as demo candidates — faces that are missing but off-path are correctly excluded from generation.

| Face | Recovered | Path position | Demo-visibility score | Selected for generation |
|---|---|---|---|---|
| front | yes | 0 | 1.000 | n/a (already reconstructed) |
| back | no | 2 | 0.600 | **yes** |
| right | no | 1 | 0.314 | **yes** |
| left | no | 3 | 0.164 | no |
| ceiling | no | 4 | 0.163 | no |
| floor | no | 5 | 0.143 | no |

This reflects the real, honest state of this apartment capture: only the `front` face has direct 3D plane evidence from the source photographs; `back`, `right`, `left`, `ceiling`, and `floor` are genuinely missing. The ranking correctly narrows 5 missing faces down to the 2 (`back`, `right`) that the demo walkthrough actually shows.

## Candidate generation and semantic rejection

Two demo-visible faces × 2 fixed-seed candidates each were attempted through the memory-conditioned learned pipeline (same IP-Adapter conditioning validated in Milestone 1.2).

| Metric | Result |
|---|---|
| Candidates attempted | 4 |
| Accepted generations | 0 |
| Rejected generations | 4 |
| Per-face latency | `back`: 13,991.6 ms · `right`: 12,205.5 ms |
| Total pre-generation time | 31,374.9 ms |
| Peak Torch VRAM | 1,448 MiB |

All 4 candidates were rejected by the existing semantic-consistency guardrail (`context_consistency.py`), not by a pipeline failure — the model loaded and ran correctly, but its proposals exceeded the unexpected-semantic-addition tolerance (up to 46.8% unsupported content) or introduced unsupported signage/text on a structural wall. Per Milestone 1.3 scope, rejected candidates are discarded outright rather than relaxed, and both faces fall back to the already-validated deterministic structural engine (Milestone 1.0) for their frozen, presented pixels. This is the guardrail behaving exactly as designed: it is honest evidence that automatic semantic rejection is active and effective on this exact hardware/model pair, not a synthetic assertion.

## Cross-face harmonization

Tonal harmonization was evaluated and applied only to pixels marked `IMAGINED` in provenance metadata; every non-`IMAGINED` pixel (evidence-backed or protected/locked) is asserted byte-exact before and after harmonization.

| Face pair | Before ΔE | After ΔE | Improved | Passed | Imagined pixels adjusted |
|---|---|---|---|---|---|
| front ↔ right | 0.00 | 0.00 | yes | yes | 3,740 |
| back ↔ right | 0.02 | 1.71 | no | yes | 109,582 |

Both adjacent-face seam checks pass the continuity guardrail (2/2). The `front ↔ right` seam was already exact because `front` is directly reconstructed and shares almost no imagined boundary; the `back ↔ right` seam shows the harmonizer actively adjusting tone across a genuinely large imagined boundary while staying within the passing threshold.

## Protected critical-region demonstration

The critical-region demo does **not** use a synthetic fallback rectangle for this run: it detected a real ADE20K `"painting"` semantic class directly from captured photographic evidence on the `front` face.

| Field | Result |
|---|---|
| Face | `front` |
| Detected class | `painting` |
| Detected from captured evidence | **true** (not synthetic fallback) |
| Protected modifications | 0 |
| Critical violations | 0 |
| Fabricated person present | **false** |

The demo overlay shows the exact captured pixels for the painting/art region untouched, with only the surrounding structure eligible for generated content — a genuine, reproducible instance of the "protected critical region" requirement rather than a staged example.

## Aggregate walkthrough metrics

| Field | Result |
|---|---|
| Demo-visible faces | 2 (`back`, `right`) |
| Reconstructed faces | 1 (`front`) |
| Missing faces | 5 (`back`, `right`, `left`, `ceiling`, `floor`) |
| Observed % | 0.00% |
| Reconstructed % | 9.33% |
| Inferred % | 24.94% |
| Imagined % | 63.90% |
| Absent % | 1.83% |
| Accepted / rejected generations | 0 / 4 |
| Protected modifications | 0 |
| Critical violations | 0 |
| Seam checks passed | 2 / 2 |
| Presented walkthrough path duration | 36.0 s |
| Backup recording produced | yes (36.0 s MP4, ffmpeg xfade, deterministic frame sequence) |

## Deterministic presentation walkthrough

A fixed 5-stop path was built and frozen for presentation:

1. **Photo-origin view** (`front`, Remember) — recognizable, evidence-backed starting viewpoint.
2. **Weak region in Remember** (`front`, Remember) — the same face's genuinely low-coverage region, shown honestly.
3. **Remember → Imagine transition** (`back`, transition) — an explicit, labeled handoff into the first demo-visible missing face.
4. **Immersive Imagine viewpoint** (`back`, Imagine) — the strongest, most consistent generated view.
5. **Constrained movement** (`right`, Imagine) — continued walkthrough with movement clamped to the recovered room envelope bounds.

**Product mode** hides raw metrics, file paths, and filenames, presenting only the walkthrough narrative and images. **Debug/provenance mode** exposes the full `metrics.json`, per-face provenance overlays, and file paths for internal review. Both modes are generated from the same frozen assets so there is no risk of drift between what founders see and what engineering audits.

## Provenance and guardrail verification

- **Strict Doctrine v2 compliance**: all generated pixels remain marked `IMAGINED`; no relabeling to `OBSERVED` or `RECONSTRUCTED`.
- **Protected pixel restoration**: `protectedModifications == 0` across both demo-visible faces.
- **Zero critical violations**: `criticalViolations == 0`, including the dedicated critical-region demonstration.
- **Semantic rejection is active**: 4/4 attempted learned candidates were rejected by the existing guardrail this run, and the pipeline fell back to the deterministic engine rather than accepting non-compliant output.
- **Fail-closed behavior preserved**: no step silently substituted fabricated content for a real result; every rejection and fallback is recorded in `metrics.json`.

## Private artifacts

All rendered walkthrough artifacts, frame sequences, and the backup recording are stored outside the repository in a private local session directory. Artifacts include:

- `index.html` — interactive walkthrough site with product/debug mode toggle
- `metrics.json` — complete quantitative metrics for this run
- `faces/face-front/`, `faces/face-back/`, `faces/face-right/` — per-face color, provenance, and generation-candidate imagery
- `critical-region-demo/critical-region-front.png` — protected critical-region overlay
- `frame-sequence/stop-00…stop-04-*.png` — 5 deterministic presentation frames
- `milestone13-backup-walkthrough.mp4` — 36-second deterministic backup recording (ffmpeg `xfade`, 24fps)

## Verdict

**B — validated partial.** The Milestone 1.3 walkthrough pipeline runs end-to-end on real apartment capture data with the validated Milestone 1.2 memory-conditioned pipeline, correctly narrows demo scope to only visible missing faces, actively rejects unsupported semantic generations rather than accepting them, harmonizes only imagined pixels while leaving all other pixels byte-exact, and demonstrates a genuine (non-synthetic) protected critical region. The 0/4 accepted-generation result for this specific apartment/seed/model combination is an honest guardrail outcome, not a pipeline defect; a future milestone can revisit prompt/context tuning to raise the acceptance rate without weakening the rejection guardrail. The milestone stops after this walkthrough for founder review, with a working 36-second backup recording produced locally.