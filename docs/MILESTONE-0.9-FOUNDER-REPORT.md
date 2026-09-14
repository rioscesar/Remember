# Founder steering update — Milestone 0.9

## Scope

Milestone 0.9 builds a local desktop 3D walkthrough over the existing
Milestone 0.8 evidence export. `tools/milestone09_walkthrough.py` recovers a
canonical room frame, fits a robust coarse envelope, detects and assigns
structural planes to a six-face box shell, projects evidence-first per-face
textures, places partial evidence-backed objects from residual (non-planar)
points, and emits a founder view plus a debug provenance view — all rendered
with pure CSS 3D transforms (no WebGL/Three.js). It does not add Android/AR
work, Gaussian tuning, COLMAP tuning, or generative completion.

This is a revision of the first Milestone 0.9 pass, which produced only a
flat 2D evidence-texture image. That was judged insufficient: it did not
give a spatial "walkthrough" and its orientation step was easy to mistake for
a gravity estimate. This revision replaces the flat image with an actual
navigable 3D room shell and documents the gravity limitation explicitly in
the metrics rather than leaving it implicit.

## What changed from the flat-texture version

- **Real 3D shell, not an image.** The room envelope is rendered as a
  six-face CSS3D box (`left/right/floor/ceiling/back/front`). Each RANSAC
  plane from `representation_spike.detect_planes` is classified by its
  normal direction and assigned to whichever face it best matches
  (`classify_face`). A face with no matching plane is rendered as an
  explicit "NO PLANE EVIDENCE RECOVERED" panel — it is never silently filled
  in or invented.
- **Camera-coherent movement.** The generated `index.html` supports
  pointer-drag look and WASD/arrow-key translation, clamped so the viewer
  cannot leave the recovered room envelope. This is ordinary CSS
  `transform-style: preserve-3d` plus `perspective`, not WebGL/Three.js.
- **Explicit gravity documentation.** `canonical_orientation()` now reports
  `gravityEstimated: false` and a `gravityBlockedReason` string
  ("No IMU/gravity sensor stream available in this offline COLMAP-only
  pipeline") directly in the metrics JSON, rather than a docstring caveat
  that a reader could miss. The orientation is PCA-derived room axes with
  camera-centroid sign resolution — useful for a consistent frame, not a
  gravity vector.
- **Partial evidence-backed object placement.** Points left over after plane
  fitting (`return_residual=True` on `detect_planes`) are grid-clustered
  (`cluster_residual_points`) into axis-aligned evidence cards
  (`build_object_cards`): measured 3D position, size, and mean color only —
  explicitly labeled "no shape completion." A compactness filter rejects any
  cluster whose bounding span exceeds 40% of the room envelope's own span on
  any axis, because grid-connectivity clustering can chain through smoothly
  varying point density into a single room-spanning "object." Rejected
  clusters are counted and reported (`rejectedRoomSpanningClusters`), not
  silently dropped.

## Doctrine and validation

Texture projection preserves OBSERVED and RECONSTRUCTED pixels per face and
routes unsupported structural gaps through `bounded_inference_fill`. A
supplied critical mask is passed to that same Doctrine v2 guardrail; critical
unsupported pixels remain ABSENT. `tools/face_guardrail_test.py` (rerun this
milestone against the `representation_spike.py` change) still reports zero
critical pixels ever inferred over.

`tools/milestone09_synthetic_test.py` uses procedural wall/floor/blob
geometry (no private photographs) and asserts: a finite canonical frame,
zero critical inference, correct face classification, non-zero residual
clustering producing a valid object card for a compact synthetic blob, and —
new this revision — that feeding the entire synthetic scene back in as
"residual" correctly triggers the room-spanning rejection filter, proving
the compactness guard is not a no-op. Latest run: `passed: true`,
`roomSpanningArtifactsRejected: 5` on the whole-scene case.

## Measured prototype result (real private apartment run)

Run against `sparse-strong/dense/fused-candidates.ply` (a PINHOLE-camera
COLMAP dense reconstruction from an earlier milestone), output written to a
private, non-committed directory:

- **Structural shell: 3 of 6 faces recovered with evidence (50% shell
  completeness).** `right`, `floor`, and `ceiling` have RANSAC-plane-backed
  evidence textures with multi-view point support (3,665 / 38,913 / 2,044
  points from 4 / 9 / 5 distinct source views respectively). `left`, `back`,
  and `front` have no matching plane and are rendered as explicit
  "NO PLANE EVIDENCE RECOVERED" panels — this capture simply does not have
  enough coherent planar evidence on those sides, and the tool says so
  instead of guessing.
- **Object placement: 2 evidence cards placed, 9 candidate clusters
  correctly rejected as room-spanning artifacts.** After tightening the
  clustering radius (density-adaptive, not a fixed fraction of a
  possibly-unrepresentative bounding diagonal) and adding the compactness
  filter, the two surviving objects have plausible, sub-room-scale local
  spans (roughly 1.28 × 1.18 × 0.43 and 0.015 × 0.05 × 0.27 units) with
  100% multi-view point support. They carry position, size, and mean color
  only — no shape completion, no invented geometry.
- **Orientation:** finite, deterministic PCA frame;
  `gravityEstimated: false`, reason recorded in the same metrics file.
- **index.html:** generated (8.2 KB), contains the six-face CSS3D shell
  markup, the object-card overlays, and the founder/debug provenance toggle.

Prototype presence is measured, not asserted: the metrics file and
`index.html` are produced by running the tool, not hand-written, and every
"no evidence" state above is an explicit label in that same file rather than
an omission.

## Founder verdict: **B — PARTIAL**

There is now a real, navigable 3D room shell (not a flat image), with
camera-coherent movement, honestly-labeled missing faces, documented gravity
limitations, and measured partial object placement with an artifact-rejection
guard proven by test. This is not an A-SUCCESS: half the shell faces have no
plane evidence in this capture, the box is a coarse symmetric approximation
(not per-plane-offset-precise), scale remains an arbitrary SfM unit (no
distance is displayed, per existing product principles), and object
placement recovered only 2 small evidence cards from one dataset. Review the
private `index.html`, toggle debug provenance, and decide whether this
bounded presence warrants the next experiment. Stop at founder review; do
not proceed to Android, AR, Gaussian tuning, or generative completion.