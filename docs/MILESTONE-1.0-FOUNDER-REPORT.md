# Founder steering update — Milestone 1.0

## Scope

Milestone 1.0 extends the Milestone 0.9 private desktop walkthrough with an
explicit **Remember / Imagine** split. The pipeline remains local and
privacy-preserving: it operates over the canonical six-face room shell and
per-face atlas textures produced from recovered geometry, not over source
photo frames. It does not use Azure, cloud inference, Android, AR, or private
artifact commits.

## Environment and generative approach selection

Inspection found an NVIDIA GeForce RTX 3070 Laptop GPU with 8 GB VRAM and a
private Python 3.10 environment containing NumPy, OpenCV, Torch
`2.1.2+cu118`, CUDA availability, Transformers, and Pillow. The available
Hugging Face cache contained segmentation models
(`nvidia/segformer-b0-finetuned-ade-512-512`,
`openmmlab/upernet-convnext-tiny`) but no Diffusers install and no cached
local image-generation model.

The smallest credible privacy-preserving approach for this machine would be a
small local inpainting/diffusion model in half precision, run only on canonical
room-face atlases with hard masks. Because the required local model/runtime was
not installed or cached, Milestone 1.0 does **not** pretend a learned generator
ran. It implements a truthful deterministic local fallback instead:

- missing room faces are completed first as generic structural atlas cards;
- generated pixels are marked **IMAGINED**, never evidence;
- the fallback is seeded only from aggregate recovered structural face colour;
- there is no source-frame generation, no cloud call, and no generated private
  artifact committed to the repository.

## Remember / Imagine pipeline

1. Canonicalize the recovered dense point cloud into the room coordinate frame
   from Milestone 0.9.
2. Detect structural planes and assign them to the six canonical faces
   (`left`, `right`, `floor`, `ceiling`, `back`, `front`).
3. For recovered faces, build atlas-space textures from fused point colour and
   visibility support. OBSERVED/RECONSTRUCTED texels are locked.
4. Apply bounded structural inference only where allowed by Doctrine v2.
5. Generate missing room faces in atlas space using the deterministic local
   structural fallback, with every generated texel labelled IMAGINED.
6. Place residual object evidence cards from measured non-planar point
   clusters. Object gap completion remains disabled because this point-only
   pipeline has no semantic object masks or confidence sufficient to safely
   complete object appearance. The code caps object completion at 1–2 gaps but
   currently reports zero generated object gaps.
7. Render the CSS3D walkthrough with:
   - **Remember view:** OBSERVED / RECONSTRUCTED / INFERRED evidence only;
   - **Imagine view:** adds IMAGINED generic structural room faces;
   - **Debug provenance:** colour-coded provenance for faces.

## Provenance and mask policy

The enforced provenance priority is:

**OBSERVED > RECONSTRUCTED > INFERRED > IMAGINED > ABSENT**

The shared Doctrine module now exposes mask/compositing helpers:

- **LOCKED:** all OBSERVED and RECONSTRUCTED texels, plus all critical regions;
- **GENERATABLE:** only currently ABSENT low-risk structural texels;
- **ABSENT:** ambiguous unsupported texels and all unsupported critical regions.

Candidate completions can only replace lower-priority provenance and can never
write through a LOCKED mask. Without per-pixel semantic critical masks in the
current point-only PLY input, Milestone 1.0 limits Imagine to generic structural
room-face atlases and leaves object completion off.

## Metrics now emitted

`tools/milestone09_walkthrough.py` now emits aggregate metrics for:

- per-face OBSERVED / RECONSTRUCTED / INFERRED / IMAGINED / ABSENT percentages;
- aggregate shell provenance percentages;
- generated room-face count and per-face generated pixel counts;
- generated object-gap count;
- deterministic generation latency;
- VRAM before/after/delta when `nvidia-smi` is available;
- shell completeness and spatial envelope consistency.

## Measured private prototype run

The updated tool was run against the existing private `sparse-strong` dense
evidence export, writing output only under the session artifact directory
(`milestone10-founder-review`). Aggregate results:

- **Remember shell:** 3 of 6 faces recovered with evidence (50% shell
  completeness): `right`, `floor`, and `ceiling`.
- **Imagine shell:** 3 missing structural faces generated in canonical atlas
  space: `left`, `back`, and `front`; each is 100% IMAGINED and visible only
  in Imagine/debug modes.
- **Aggregate face provenance:** 0.00% OBSERVED, 2.02% RECONSTRUCTED,
  12.82% INFERRED, 76.38% IMAGINED, 8.78% ABSENT across 971,520 atlas
  pixels. The zero OBSERVED value reflects this dense evidence export's
  multi-view visibility support; it does not mean no evidence was used.
- **Objects:** 2 measured residual evidence cards placed; generated object
  gaps remain `0`; 9 room-spanning residual clusters rejected.
- **Generation performance:** deterministic Imagine generation latency
  measured at about 2.9 seconds on the final validation run; `nvidia-smi` reported 243 MB VRAM before and
  after generation (`0` MB delta), consistent with the non-learned fallback.
- **Verdict emitted by metrics:** `B-PARTIAL`.

## Validation

Synthetic tests use procedural geometry only; no private photographs or
generated private artifacts are used.

Latest local validation:

```text
<private-python> tools\milestone09_synthetic_test.py
<private-python> tools\face_guardrail_test.py
```

Results:

- zero observed/reconstructed modification by lower-priority candidates;
- zero critical overwrite;
- generated missing room faces counted before any object-gap consideration;
- generated object gaps: `0`;
- aggregate provenance percentages sum to 100%;
- spatial face transforms are preserved through Imagine generation;
- existing critical guardrail still reports `criticalPixelsEverInferredOver: 0`.

## Founder verdict: **B — PARTIAL**

Milestone 1.0 is a credible partial founder-review build: the walkthrough now
has a clear Remember/Imagine toggle, atlas-space missing-room-face completion,
hard provenance priority, locked critical/evidence masks, debug provenance, and
synthetic proof that critical and observed/reconstructed regions are not
overwritten.

It is not an A-success because no local learned generative model was available
in the inspected environment, so Imagine uses a deterministic generic
structural fallback rather than semantic image generation; object gaps are not
completed; and missing faces remain explicitly IMAGINED rather than evidence.
Stop at founder review. Do not proceed to Android/AR.
