# Founder steering update — Milestone 0.9

## Scope

Milestone 0.9 adds a local desktop walkthrough over the existing Milestone 0.8
evidence export. `tools/milestone09_walkthrough.py` recovers a canonical room
frame, fits a robust coarse envelope, projects coloured points into an
evidence-first atlas, and emits a founder view plus a provenance view. It does
not add Android/AR work, Gaussian tuning, COLMAP tuning, or generative
completion.

The canonical frame is deterministic: robust PCA supplies the axes and the
camera-centroid sign resolves the front direction. The envelope uses the
2nd–98th percentile bounds, so isolated dense outliers cannot define the
room. Object placement is intentionally partial-evidence-only; without a
semantic mask and multi-view support the builder places zero objects rather
than inventing appearance.

## Doctrine and validation

Texture projection preserves OBSERVED and RECONSTRUCTED pixels and routes
unsupported structural gaps through `bounded_inference_fill`. A supplied
critical mask is passed to that same Doctrine v2 guardrail; critical
unsupported pixels remain ABSENT. `tools/milestone09_synthetic_test.py` uses
procedural geometry and asserts zero critical inference, finite orientation,
and non-empty multi-view reconstruction. No private photograph or output is
stored in Git.

## Measured prototype result

The prototype is present when its metrics contain non-zero
`observedOrReconstructedPercent` and the generated `index.html` offers both
founder and debug provenance views. The synthetic fixture produces a finite
canonical frame, a non-degenerate envelope, and multi-view texture pixels.
The real private apartment run remains an input-dependent local command; its
aggregate metrics are emitted beside the private prototype and are not copied
here.

## Founder verdict: **B — PARTIAL**

There is a measured, auditable walkthrough surface and canonical orientation,
but it is not an A-SUCCESS claim: coarse envelope planes and evidence cards
are not yet a full room reconstruction, and no general object placement path
is enabled without semantic evidence. Review `index.html` from a private run,
toggle debug provenance, and decide whether this bounded presence warrants the
next experiment. Stop at founder review; do not proceed to Android, AR,
Gaussian tuning, or generative completion.
