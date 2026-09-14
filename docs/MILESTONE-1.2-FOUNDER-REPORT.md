# Founder steering update — Milestone 1.2

Milestone 1.2 evaluated local visual memory conditioning for canonical room-face atlas completion using local IP-Adapter conditioned on ranked source photographs. It produced a controlled 4-way comparison for the exact `back` room face: Remember (input evidence), deterministic Imagine (Milestone 1.0), unconditioned learned Imagine (Milestone 1.1A), and memory-conditioned learned Imagine (Milestone 1.2). The pipeline ran strictly locally on the RTX 3070 8GB, verified context consistency, preserved hard mask and provenance guardrails, and stopped after this single comparison for founder review. No private photographs, models, or comparison artifacts were added to Git.

## Runtime, model, and visual conditioning

| Item | Result |
|---|---|
| Python executable | `C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\radiance-env\python310\python.exe` |
| Python / Torch / CUDA | 3.10.11 / 2.1.2+cu118 / CUDA 11.8 |
| GPU / total VRAM | NVIDIA GeForce RTX 3070 Laptop GPU / 8,191.5 MiB |
| Base Inpainting Model | `stable-diffusion-v1-5/stable-diffusion-inpainting` (fp16) |
| Visual Conditioning | `h94/IP-Adapter` (`ip-adapter_sd15.safetensors`, fp16, scale `0.7`) |
| Semantic Model | `openmmlab/upernet-convnext-tiny` (ADE20K semantic segmentation) |
| Pipeline Space | Canonical room-face atlas space (not frame-by-frame) |

## Context source photo ranking

Source photographs were evaluated and ranked using 4 multi-modal signals:
1. **Camera-to-target geometry & 3D point support**: Multi-view 3D points visible in camera and belonging to structural planes.
2. **Spatial proximity**: Euclidean distance of camera center to room origin.
3. **Ray alignment**: Dot product between camera view ray and target face normal.
4. **Semantic wall area fraction**: Wall surface coverage detected in the photo.

Top 4 ranked photographs for the `back` face:
1. `20260709_193049.jpg` (composite score: **0.7583**, wall points: 40,145, distance: 7.12 m, alignment: 0.575)
2. `20260709_192437.jpg` (composite score: **0.6618**, wall points: 28,864, distance: 4.56 m, alignment: 0.568)
3. `20260709_192425.jpg` (composite score: **0.6556**, wall points: 27,935, distance: 5.21 m, alignment: 0.655)
4. `20260709_192540.jpg` (composite score: **0.5326**, wall points: 18,273, distance: 6.50 m, alignment: 0.660)

A private context contact sheet composite was generated and saved to the private session folder.

## Controlled 4-Way comparison (`back` face)

The exact `back` face was evaluated across all 4 conditions with fixed seed (`1101`), 24 steps, guidance scale `6.0`, and exact protected pixel restoration:

| Metric | Condition 1: Remember | Condition 2: Deterministic Imagine | Condition 3: Unconditioned Imagine (M1.1A) | Condition 4: Memory-Conditioned (M1.2) |
|---|---|---|---|---|
| **Description** | Direct input evidence only | Milestone 1.0 structural gradient | Milestone 1.1A SD 1.5 inpainting | Milestone 1.2 SD 1.5 + IP-Adapter |
| **Observed/Reconstructed** | 12.06% | 0.00% (masked base) | 12.06% | 12.06% |
| **Imagined Pixels** | 0.00% | 87.94% | 87.94% (606,900 px) | 87.94% (606,900 px) |
| **CIE Lab Delta E** | N/A (Ground Truth) | 58.20 | 40.61 | **34.80** (improved palette fidelity) |
| **Color Similarity Score** | N/A | 41.80 | 57.74 | **65.87** |
| **Structural Edge Score** | N/A | 72.10 | 98.71 | **99.24** |
| **ADE20K Structural %** | 100.0% | 100.0% | 88.23% (building/wall) | **85.19%** (wall/floor/ceiling) |
| **Unexpected Additions %**| 0.00% | 0.00% | 11.77% (outdoor road/plants) | 14.81% (interior door/window) |
| **Protected Modifications**| 0 | 0 | **0** | **0** |
| **Critical Violations** | 0 | 0 | **0** | **0** |
| **Peak Torch VRAM** | 0 MiB | 0 MiB | 2,650 MiB | **3,704 MiB** |
| **Generation Latency** | < 1 ms | 4.2 ms | 8,837.9 ms | **7,356.9 ms** |

## Provenance and guardrail verification

- **Strict Doctrine v2 Compliance**: Contextual source photos guided style and color palette, but all generated pixels were marked strictly **`IMAGINED`** in provenance metadata (zero relabeling as `OBSERVED` or `RECONSTRUCTED`).
- **Protected Pixel Restoration**: In both learned conditions, 100% of non-generatable and locked pixels were restored exactly from base evidence (`protectedModifiedPixels == 0`).
- **Zero Critical Violations**: `criticalViolations == 0`.
- **Fail-Closed Behavior**: If local weights or IP-Adapter checkpoints are missing or invalid, the pipeline fails closed cleanly and reports the exact blocker rather than fabricating fake output.

## Private artifacts

All rendered comparison artifacts and contact sheets are stored locally in the private session directory:
`C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\milestone12-controlled-comparison`

Artifacts include:
- `index.html`: Interactive 4-way visual comparison sheet
- `context-contact-sheet.png`: 4-photo ranked visual reference sheet
- `condition1-remember.png` & `condition1-remember-provenance.png`
- `condition2-deterministic.png` & `condition2-deterministic-provenance.png`
- `condition3-unconditioned.png` & `condition3-unconditioned-provenance.png`
- `condition4-conditioned.png` & `condition4-conditioned-provenance.png`
- `metrics.json`: Complete quantitative metrics and consistency scores

## Verdict

**B — validated partial.** Local IP-Adapter visual memory conditioning was successfully validated on the RTX 3070 8GB within ~3.7 GB VRAM. It delivered measurably improved color and palette fidelity (Delta E reduced from 40.61 to 34.80; color similarity increased from 57.74 to 65.87) and high edge continuity (99.24), while strictly obeying Doctrine v2 provenance and exact protected pixel preservation. The milestone stops immediately after this single face comparison for founder review.