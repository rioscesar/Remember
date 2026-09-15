# Founder steering update -- Milestone 1.5

Milestone 1.5 replaces the Milestone 1.0-1.4 generated-face "hero" cube-face
walkthrough with a **photo-anchored spatial corridor**: two REAL captured
photographs are the hero anchors, shown at full byte-exact fidelity and split
into depth-aware multiplane parallax layers built only from actually
photographed pixels, connected by a deliberately **subordinate** connective
shell that is desaturated, blurred, lower-resolution, reduced-opacity, and
always labelled as non-photographic connective tissue -- never confused with
the hero evidence. This is a raw working-demo build (10-20s autoplay,
screenshots, no CSP hardening, no product/debug mode toggle, no reset/epoch
machinery), not a polished/frozen presentation.

## What changed vs. Milestones 1.0-1.4

| | Milestones 1.0-1.4 (generated face hero) | Milestone 1.5 (photo-anchored corridor) |
|---|---|---|
| Hero content | Generated/inpainted room-cube faces (`back`, `right`, ...) | Two real captured photographs, byte-exact, never regenerated |
| Depth | Flat 2D atlas per face | Depth-aware multiplane parallax (near/mid/far layers from real 3D evidence) |
| Connective space | Support-gated radiance bridge or hard crossfade between anchors | Explicitly subordinate connective shell (opacity 0.55, half-resolution, blurred, desaturated, labelled IMAGINED) |
| Presentation | Frozen, CSP-hardened, product/debug toggle, reset/epoch guard (Milestone 1.4) | Raw: single autoplay path, metrics printed inline, no hardening |

## Reused, unmodified building blocks

- **Camera poses + spatial graph** -- `spatial_graph.py`'s `graph.json` (nodes carry recovered camera center/quaternion; edges carry the strong/weak evidence classification). `select_hero_pair` uses it directly to choose hero A (highest strong-edge evidence weight) and hero B (the other end of hero A's single best edge).
- **Provenance/critical masks** -- `evidence_doctrine.py` (`OBSERVED`/`RECONSTRUCTED`/`INFERRED`/`IMAGINED`/`ABSENT`, `generation_masks`, `bounded_inference_fill`, `apply_provenance_priority`, `provenance_percentages`) unchanged.
- **Room envelope** -- `milestone09_walkthrough.canonical_orientation` / `fit_room_envelope`, used to report/clamp corridor extent; hero photographs are never reprojected into a room atlas.
- **Constrained refinement** -- `learned_inpainting.run_learned_inpainting` at low strength (`0.35`), used only on the subordinate shell canvas via the existing `runner` test seam, failing closed to the deterministic shell exactly like every earlier milestone's learned engine.

## New Doctrine v2 guarantee: hero photo fidelity

Every non-transparent pixel of every hero depth layer is asserted **byte-exact**
to the original captured photograph -- this is checked directly against the
production code path in `milestone15_synthetic_test.py`
(`test_depth_layers_preserve_photo_fidelity`), not re-implemented. Hero pixels
are never touched by any generation step; `criticalViolations` is always `0`
by construction because no critical region is ever painted over on a hero.
The connective shell is a pure synthetic canvas (100% `IMAGINED`, `0%`
`OBSERVED`/`RECONSTRUCTED`) and is always rendered at lower opacity (`0.55`
vs. hero's `1.0`) and lower resolution (`0.5x` vs. hero's native resolution)
so it can never visually compete with a hero.

## Depth-aware multiplane parallax

`build_depth_layers` projects the reused 3D evidence points (from
`export_dense_evidence.py`'s dense JSON) into each hero photo using its
recovered camera pose, builds a nearest-point-wins sparse depth buffer, and
extends it with the existing `bounded_inference_fill` (the same classical,
non-learned, distance-capped inpaint used everywhere else in this codebase)
to close small evidence gaps -- pixels beyond the bound are conservatively
left "flat" (no parallax shift, `ABSENT` depth provenance) rather than
guessed. The classified depth is then bucketed into N layers (default 3);
each layer's RGB is a direct copy of the hero photo restricted to its mask,
so **every pixel of the real photo remains visible in exactly one layer** --
no synthesized geometry, no holes.

## Validation

Ran on this machine (`Python.Python.3.12`, installed via `winget` for this
session; `numpy`, `opencv-python-headless`):

| Suite | Result |
|---|---|
| `tools/milestone15_synthetic_test.py` (new, private-data-free) | **9/9 passed** |
| `tools/face_guardrail_test.py` | passed (unchanged) |
| `tools/milestone09_synthetic_test.py` | passed (unchanged) |
| `tools/milestone13_synthetic_test.py` | passed (unchanged) |
| `tools/milestone14_synthetic_test.py` | passed (unchanged) |
| `tools/milestone13a_synthetic_test.py` | not runnable here (`ModuleNotFoundError: torch`, pre-existing optional dependency, unrelated to this change) |

The new suite covers: hero-pair selection over the reused spatial graph
(strongest-evidence default, override, edgeless-graph rejection);
outside-Git output enforcement (same pattern as `milestone14_demo_freeze`);
hero photo byte-exact fidelity across every depth layer with full-photo
coverage (no holes); rejection when no 3D evidence projects into a hero
photo; connective-shell subordination (opacity/resolution) and 100%
`IMAGINED` provenance; the learned-refinement fail-closed path *and* the
runner-seam accepted path (still 100% `IMAGINED`); parallax screenshot
compositing actually shifts pixels; and raw-walkthrough duration clamping
to 10-20s plus the absence of Milestone 1.4-style CSP/product-debug/epoch
polish in the generated HTML.

## Raw artifacts produced this run

No real apartment capture set for this repository's photo-corridor pipeline
was available in this environment, so the end-to-end builder
(`run_photo_corridor`) was exercised on a small procedurally generated
fixture (two synthetic "hero" photographs, a 3-node spatial graph with one
strong and one weak edge, and ~12k synthetic 3D evidence points) to produce
a genuine, working raw build -- the same code path a founder would point at
real photographs, `graph.json`, and dense-evidence JSON.

**Private raw output (outside Git, not committed):**
`C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\milestone15-demo\raw-output\`

Contents: `index.html` (raw autoplay corridor), `metrics.json`, `heroes/*.png`
(6 depth-layer PNGs, 3 per hero, byte-exact hero pixels only),
`shell/shell.png` (subordinate connective canvas), `screenshots/*.png` (8 raw
parallax-offset screenshots, 4 per hero). Fixture inputs (`graph.json`,
`dense-evidence.json`, `photos/`, `build_demo.py`) live alongside it in
`...\files\milestone15-demo\`.

Measured on this run: hero depth provenance ~4.5% directly `OBSERVED` from
projected 3D evidence and ~95.5% `INFERRED` (bounded fill) with `0%` left
flat/`ABSENT` -- expected for a sparse synthetic point cloud rather than a
dense real capture; a real dense-evidence export would raise the `OBSERVED`
share substantially. Shell: `100%` `IMAGINED`, `0` critical violations,
opacity `0.55`, resolution `0.5x`. Duration clamped to `15.0s` (within the
required 10-20s raw-walkthrough window), `8` screenshots produced.

## Scope boundaries respected

No Android/AR, no model swap, no semantic-threshold change, no relaxing of
Representation Doctrine v2, no change to `evidence_doctrine.py`,
`spatial_graph.py`, `learned_inpainting.py`'s validation logic, or
`milestone09_walkthrough.py`'s room-envelope math -- all are reused as-is.
Only `tools/milestone15_photo_corridor.py` (new builder),
`tools/milestone15_synthetic_test.py` (new tests), and this report were
added. No photograph, render, mask, model file, or private path was added to
Git.

## Verdict

Ready for founder review. Work stops here; nothing was pushed or committed.
