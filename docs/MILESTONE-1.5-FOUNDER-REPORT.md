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


---

# Addendum: Milestone 1.5A -- real apartment capture integration

Milestone 1.5's builder (`tools/milestone15_photo_corridor.py`) has now been
exercised end-to-end against a genuine apartment capture set (real
photographs, real COLMAP camera poses/intrinsics, real dense 3D evidence,
real per-photo critical masks) instead of the synthetic fixture above. No
capture-specific path, filename, or constant was added to Git; only
generalized reader/render functions were added to the tool.

## What was added (generalized, capture-agnostic)

- **Real COLMAP camera intrinsics.** `representation_spike.parse_camera`
  strictly requires the PINHOLE model. Real capture pipelines commonly
  register `SIMPLE_RADIAL`/`SIMPLE_PINHOLE`/`RADIAL`, one distinct camera per
  photo. `parse_colmap_cameras_txt` / `parse_colmap_image_camera_ids` /
  `load_real_camera_intrinsics` read these directly (radial distortion is
  not applied -- treated as a pinhole approximation, exactly like the
  existing heuristic, except now driven by the real recovered focal
  length). `scale_camera_to_photo` rescales intrinsics when the hero photo
  on disk is a downscaled preview of the calibration resolution.
- **Critical-mask threading.** `build_depth_layers` now accepts an optional
  real `critical_mask` (e.g. from `scene_risk_segmentation.py`'s `.npz`
  export via the new `load_critical_mask_npz`), reported in depth
  provenance/critical-pixel bookkeeping. It never changes which real pixels
  are shown -- verified by a synthetic test that the union of visible pixels
  stays byte-exact with or without a mask.
- **Provenance screenshot.** `render_provenance_screenshot` produces a raw
  overlay (`evidence_doctrine.py` codes only -- never conflated with any
  other module's provenance legend) tinting OBSERVED/INFERRED/ABSENT depth
  regions and outlining any critical mask, without altering the underlying
  photo pixels.
- **Raw recording.** `render_raw_recording` concatenates the existing raw
  screenshot frames into a single 10-20s video via `ffmpeg` (hard cuts only,
  no crossfade -- deliberately rawer than Milestone 1.3's crossfaded backup
  recording pattern it's modeled on), failing closed to the existing PNG
  frame sequence with an explicit blocker message if `ffmpeg` is
  unavailable.
- **Pose override for cross-reconstruction coordinate frames.** The
  real-capture run surfaced an important finding: the reused spatial graph
  and a given dense-evidence export can come from *different* reconstruction
  runs (different registered-photo counts, different coordinate
  frame/scale) even when they share photo names. Projecting a graph node's
  pose into a dense-evidence run it did not come from produced zero
  supported pixels. `load_pose_overrides` (via
  `representation_spike.parse_views`, reused unmodified) reads poses
  straight from the SAME `images.txt` as the dense evidence; `pose_overrides`
  on `run_photo_corridor` substitutes a hero's graph-node pose with the
  matching override when present. The graph still decides which two photos
  are heroes (edge/evidence classification is unaffected); only the pose
  *numbers* used for projection are corrected to the matching reconstruction.

## Validation

`tools/milestone15_synthetic_test.py` grew from 9 to 16 tests (all still
private-data-free): real-shaped COLMAP camera/pose parsing (SIMPLE_RADIAL
and PINHOLE), intrinsics rescaling, critical-mask threading without altering
shown pixels, the provenance screenshot, the pose-override end-to-end path
(including a synthetic reproduction of the coordinate-frame mismatch), and
the raw-recording ffmpeg/fallback behavior. **16/16 passed.** The full
existing regression set (`face_guardrail_test.py`,
`milestone09_synthetic_test.py`, `milestone13_synthetic_test.py`,
`milestone14_synthetic_test.py`) still passes unchanged.

An end-to-end run against the real apartment capture set (outside this
repository, never committed) produced all seven requested raw artifacts:
a byte-exact hero-A reference photo, an entry screenshot, a mid-corridor
(subordinate shell) screenshot, a byte-exact hero-B destination photo plus
destination screenshot, a real `ffmpeg`-produced ~16.7s raw hard-cut MP4
recording (within the 10-20s requirement), a provenance screenshot per hero,
and `metrics.json`. Both hero reference photos were independently
SHA-256-verified byte-identical to the original captured files. Depth
provenance on this real, sparse dense-evidence set was low
single-digit-percent directly `OBSERVED` (expected for a sparse point
export against full-resolution photo pixel counts) with the remainder
bounded-`INFERRED` or left flat/`ABSENT` -- every hero pixel remained
visible and byte-exact regardless, per the existing fidelity guarantee.

Nothing from this run (photos, poses, masks, dense evidence, or the private
output directory) was added to Git; only the generalized reader/render code
above and its synthetic tests were.

---

# Addendum: Milestone 1.5B -- continuous multi-hop corridor

Milestone 1.5/1.5A produced a single hero-A -> hero-B jump (a hard cut or a
short bridged/crossfaded transition between exactly two photographs).
Milestone 1.5B replaces that jump with a **continuous walkthrough** that can
route through an intermediate real photograph (A -> optional C -> B) when
the reused spatial graph shows that path is a better fit than the direct
edge, and renders it as one unbroken, no-hard-cut video sweep rather than a
sequence of discrete screenshots. No reconstruction, pose recovery, or
photo-corridor doctrine was redone or changed -- this is purely a path
selection + continuous-rendering layer on top of 1.5/1.5A's existing
reused building blocks (spatial graph, camera poses, critical masks, depth
layers, evidence doctrine).

## What was added (generalized, capture-agnostic)

- **Multi-hop path search (`select_corridor_path`).** Enumerates every
  simple path between hero A and hero B of length <= 3 nodes (direct edge,
  or via exactly one intermediate node C) using the existing `graph.json`
  edges, scores each path by a per-edge cost that rewards low angular
  change and high shared-landmark/support evidence
  (`edge_cost = angular/180 - 0.5*support - 0.5*(shared/(shared+50))`,
  averaged over the path's edges), and selects the lowest-cost path. The
  direct strong edge remains a candidate and wins whenever it is genuinely
  the better path; a via-node is only selected when the graph's own
  evidence says the two shorter hops are collectively a better fit than the
  one long hop.
- **Continuous pose interpolation.** `slerp_quaternion` / `interpolate_pose`
  (numpy-only, no new dependency) smoothly interpolate the reused recovered
  camera center/quaternion between path nodes so a single global
  path-progress value in [0, 1] maps continuously across every segment of
  the selected path with no discontinuity at a via-node.
- **Rigid-critical parallax (`composite_parallax_frame_rigid`).** Extends
  1.5/1.5A's per-hero parallax compositor so that any pixel inside a reused
  critical mask is forced back to its exact original photographed value on
  every single frame -- never shifted, never blended away -- while
  non-critical depth layers still parallax normally.
- **Support-aware blend (`provenance_confidence` /
  `support_aware_blend_weight`).** The crossfade weight between a segment's
  two endpoint photos is biased per-pixel by each side's real depth-evidence
  confidence (`OBSERVED` > `INFERRED` > `ABSENT`), so a photo's own
  strongly-evidenced regions stay more visible longer into the transition,
  while still resolving to a pure, unblended photo at each segment's start
  and end.
- **Non-gray connective space (`load_bridge_frames` /
  `build_segment_connective_frame` / `photo_crossfade_canvas`).** For any
  segment whose edge has a precomputed real evidence-gated radiance-bridge
  asset available, those real bridge frames are reused directly. For any
  segment without one, a real-photo crossfade of that segment's own two
  endpoint photographs (blurred/desaturated via the existing
  `make_subordinate`) is used instead -- still real captured pixels, never a
  flat synthetic gradient. Every segment's connective source is recorded
  explicitly in `metrics.json` (`segmentConnectiveSources`), so which real
  material backed each part of the video is always honestly disclosed.
- **Continuous no-hard-cut video (`render_continuous_video`).** Renders a
  dense sequence of frames sweeping continuously across the whole selected
  path (not per-segment, so there is no jump even at a via-node) and encodes
  them with a single ffmpeg image-sequence pass -- no `concat` demuxer, no
  hard per-frame holds, no text or debug overlay burned in. Fails closed to
  the retained PNG frame sequence if `ffmpeg` is unavailable.
- **Performance: bounded continuous-video render resolution
  (`scale_node_record_for_video`, `--video-max-dim`, default 1600px longest
  side).** The anchor reference photos, the 0/25/50/75/100% path-progress
  samples, and the provenance snapshot always render at full native photo
  resolution with byte-exact hero fidelity, exactly as in 1.5/1.5A. Only the
  thin per-frame compositing done for the hundreds of frames of the
  continuous video itself is bounded to a fixed working resolution before
  final blur/blend (the connective shell was already intentionally
  blurred/subordinate, so this changes no visible doctrine, only render
  cost).

## Validation

`tools/milestone15b_synthetic_test.py` (new, 20 tests, private-data-free):
path-cost scoring and via-node vs. direct-edge selection on synthetic
3-node graphs built both ways; pose slerp/interpolation continuity; rigid
critical-pixel preservation under parallax; support-aware blend weight
behavior at segment boundaries and mid-transition; bridge-frame loading and
fallback to real-photo crossfade when no bridge asset exists for an edge;
end-to-end `run_continuous_corridor` producing every required raw artifact
against a synthetic fixture, plus the outside-Git output-path rejection
check (same pattern as every earlier milestone). **20/20 passed.** The full
existing regression set (`milestone15_synthetic_test.py`,
`face_guardrail_test.py`, `milestone09_synthetic_test.py`,
`milestone13_synthetic_test.py`, `milestone14_synthetic_test.py`) still
passes unchanged.

## Real run (outside this repository, never committed)

Run against the real apartment capture set's registered photographs, real
COLMAP camera poses/intrinsics, real dense 3D evidence, real per-photo
critical masks, and the real evidence-gated radiance-bridge asset reused
from Milestone 1.5A. The path search selected the two-hop via-node path
(mean edge cost lower than the direct edge, by real angular-change and
shared-landmark/support evidence in the reused graph) rather than the
direct strong edge -- a genuine outcome of the new path-search feature, not
a fixture artifact. The direct edge's own precomputed radiance-bridge
frames therefore did not cover either selected segment, so both segments
correctly fell back to the real-photo crossfade connective source; this is
disclosed explicitly per-sample in `metrics.json` rather than silently
substituted.

Produced: three byte-exact anchor reference photographs (hero A, via node
C, hero B; SHA-256-verified against the originals), five path-progress
samples at 0/25/50/75/100%, a combined provenance snapshot plus one
per-node provenance overlay, `metrics.json` (selected path, per-edge
angular/support/shared-landmark scores, per-node depth-provenance and
critical-pixel percentages, connective-source-per-sample, 0 critical
violations), and one continuous, no-hard-cut, 15.0s/24fps MP4 (1600x1200
render resolution for the continuous sweep; anchors/samples/provenance
remain full native photo resolution) produced via ffmpeg. Depth provenance
across the three real path nodes was low single-digit-percent directly
`OBSERVED` (expected for this sparse dense-evidence export against full
photo pixel counts) with the remainder bounded-`INFERRED` or left
flat/`ABSENT`; every hero/via-node pixel remained visible and byte-exact
regardless, per the existing fidelity guarantee. Critical-region
rigid-preservation held at 0 violations throughout the full 360-frame
render.

## Scope boundaries respected

No reconstruction was redone, no pose was re-recovered, no representation
doctrine was pivoted or relaxed, no change to `evidence_doctrine.py`,
`spatial_graph.py`, or `milestone15_photo_corridor.py`'s existing
single-hop functions (all reused unmodified via import). Only
`tools/milestone15b_continuous_corridor.py` (new module),
`tools/milestone15b_synthetic_test.py` (new tests), and this addendum were
added. No photograph, render, mask, model file, bridge asset, or private
path was added to Git.

## Verdict

Ready for founder review. Work stops here; nothing was pushed or committed
without explicit instruction.
