# Architecture

Remember separates private reconstruction from private viewing.

## Components

| Component | Responsibility | Media boundary |
|---|---|---|
| Android app | Memory library, source URI references, bundle validation, and evidence-backed viewing | Reads selected media through Android's document provider; has no `INTERNET` permission. |
| Remember Companion | Local COLMAP Structure-from-Motion and representation evidence export | Runs on the person's computer; input and temporary output stay local. |
| Reconstruction bundle | Versioned data carrying camera poses, quality values, geometry, and source-photo names | Imported manually from local storage; contains no generated scene completion. |

## Android data flow

1. The person names a memory and chooses at least three photos through the system document picker.
2. The app retains read-only URI permissions and display names; it does not copy or alter the originals.
3. The Companion reconstructs a sparse point cloud from matching evidence and exports `reconstruction.json`.
4. The app verifies the bundle format, minimum evidence, and connection to the selected photo names before persisting it.
5. The scene view projects the imported three-dimensional landmarks in response to touch rotation. Point opacity decreases with observed support and reprojection quality.

No distance is displayed: ordinary-photo SfM has arbitrary scale unless separate measured evidence is introduced.

## Representation direction

The physical-device dense spike established that valid coloured points do not make an ordinary interior recognizable when reliable depth covers only a small fraction of its surfaces. Camera poses and correspondences remain useful. A spatial photograph graph is retained as a navigation and fallback primitive, but it has not been accepted as the primary product representation.

Semantic classification can identify enough source pixels belonging to one recovered structural wall to create a mostly covered photographic plane without requiring per-pixel reconstructed depth. The follow-on coherence spike fixed source ownership in canonical wall coordinates and preserved every contributor, eliminating ownership flicker during a small camera sweep. Photometric normalization reduced overlap disagreement, but exposure regions, semantic holes, and fragmented boundaries remained perceptually obvious. The wall therefore did not pass the continuous-photograph criterion and has not been promoted into the Android representation.

The provenance classes are:

- **Captured:** direct source pixels;
- **Reconstructed:** geometry supported by multiple observations;
- **Interpolated:** captured pixels geometrically reprojected between known viewpoints;
- **Imagined:** content without sufficient photographic evidence.

Captured, reconstructed, and quality-gated interpolated evidence are allowed. Imagined content is not part of the evidence-backed experience. See [the representation spike](REPRESENTATION-SPIKE.md), [semantic-plane spike](SEMANTIC-PLANE-SPIKE.md), [photographic-coherence spike](WALL-COHERENCE-SPIKE.md), [radiance-field representation spike](RECONSTRUCTION-REPRESENTATION-SPIKE.md), [sparse-view regularization / support-gate spike](SPARSE-VIEW-SUPPORT-GATE-SPIKE.md), [hybrid supported spatial memory prototype](HYBRID-SPATIAL-MEMORY-SPIKE.md), and [spatial-continuity photo transitions](SPATIAL-TRANSITION-SPIKE.md) for the measured alternatives and stop/go decisions.

Milestone 1.0 extends the private desktop-only 3D walkthrough
(`tools/milestone09_walkthrough.py`). It canonicalises orientation from
recovered geometry (explicitly *not* a gravity estimate -- there is no
IMU/gravity-sensor stream in this offline COLMAP-only pipeline, and the
metrics record that limitation rather than implying it), fits a robust
percentile room envelope, and renders it as a real six-face box shell using
pure CSS 3D transforms (no WebGL/Three.js). Detected planes are assigned to
whichever coarse face (left/right/floor/ceiling/back/front) their normal and
extent best match. A face with no matching plane is no longer hidden inside a
flat "no evidence" state: the prototype exposes a **Remember / Imagine**
toggle. Remember shows only evidence-backed OBSERVED, RECONSTRUCTED, and
bounded INFERRED face atlases. Milestone 1.1 generalizes the Imagine step:
the default `auto` mode targets the local
`stable-diffusion-v1-5/stable-diffusion-inpainting` Diffusers pipeline on the
strongest missing canonical room-face atlas first, using fixed seed, boring
structural prompts, hard GENERATABLE/LOCKED/CRITICAL masks, exact protected
pixel restoration after generation, and required zero protected-pixel changes
and zero critical violations before accepting any IMAGINED pixels. If
dependencies, model weights, auth/network/cache, CUDA, or guardrail validation
fail, the learned path is not presented as having run; metrics record the
blocker and the Milestone 1.0 deterministic generic structural fallback
remains available. Learned output is local-only, never source-frame
generation, and never promoted to evidence.
Residual (non-planar) points are grid-clustered into axis-aligned evidence
cards for partial object placement; a compactness filter rejects any cluster
whose span exceeds 40% of the room envelope on any axis as a clustering
artifact rather than plausible furniture, and reports the rejection count
instead of silently dropping it. Object gap completion remains disabled unless
semantic confidence is adequate. The generated `index.html` supports
pointer-drag look and WASD/arrow-key translation clamped to the envelope,
with Remember, Imagine, and debug provenance views sharing the same per-face
and per-object arrays; critical masks are still passed to Doctrine v2 and
cannot be inferred or imagined over. This is a coarse box-shell prototype
review surface, not an Android/AR scene format or a measured-scale
reconstruction, and its outputs remain private.

Milestone 0.8 introduces **Representation Doctrine v2** (`tools/evidence_doctrine.py`), which amends this for low-risk structural surfaces only: OBSERVED and RECONSTRUCTED pixels are unchanged, but a bounded, non-generative INFERRED tier is now permitted for conservative structural continuation (wall/floor/ceiling gaps), strictly gated so it can never touch identity-critical content (faces, artwork, screens, signage, mirrors, personal objects), which stays absent rather than completed. Milestone 1.0 adds explicit IMAGINED provenance for the separate founder-review Imagine layer only, and Milestone 1.1 adds optional learned atlas inpainting behind the same doctrine gates. The enforced priority is OBSERVED > RECONSTRUCTED > INFERRED > IMAGINED > ABSENT; OBSERVED/RECONSTRUCTED and critical masks are LOCKED, only ABSENT low-risk structural texels are GENERATABLE, and ambiguous or critical unknowns stay ABSENT. See [evidence-maximizing structural completion](EVIDENCE-MAXIMIZING-RECONSTRUCTION-SPIKE.md), [the Milestone 1.0 founder report](MILESTONE-1.0-FOUNDER-REPORT.md), and [the Milestone 1.1 founder report](MILESTONE-1.1-FOUNDER-REPORT.md) for the implementation, guardrail tests, and measured results.

## Evidence contract

An accepted point has coordinates, observed RGB color, and at least three supporting photographs. Sparse points also carry a non-negative reprojection error; dense geometric-fusion points explicitly have no sparse residual. Bundles carry aggregate pre- and post-filter point counts plus a quality-gate report calculated from recovered camera centers and retained geometry.

The Companion rejects a reconstruction before writing an importable bundle unless all of these objective conditions hold:

- At least three photographs are registered.
- At least 100 landmarks survive the three-observation evidence threshold.
- At least 50% of the selected photographs belong to the accepted recovered component.
- The median pairwise recovered-camera baseline is at least 0.001 reconstruction units.
- The 5th-to-95th-percentile landmark-cloud diagonal is at least half of the median camera baseline.

The Android importer independently requires `qualityGatePassed` from that report. It renders no surfaces, inferred depths, or generated regions. The background and attenuated points form the evidence boundary rather than a fabricated room.

## Dense evidence boundary

The dense spike consumes only a sparse component that has already passed the quality gate. COLMAP geometric fusion supplies candidate points, then the Companion requires at least three **distinct** source views per retained point, fully parses the visibility sidecar, preserves original fused coordinates and colors, applies the same camera-baseline-to-robust-spread gate, and additionally rejects geometry whose thinnest principal spread is under 5% of its widest, because that describes a single photographed surface rather than a place. It creates no mesh, filled surface, completion, inferred texture, or generated region. On-device SfM and AR remain deliberately separate follow-on spikes and cannot replace the provenance checks in the import contract.

## Rendering axes

COLMAP world coordinates are X right, Y down, Z forward. The scene view draws Y upward, so imported points are converted with a 180-degree rotation about X, `(x, y, z) -> (x, -y, -z)`. This is a proper rotation and preserves handedness; negating Y alone would render a mirrored scene. An earlier build omitted this conversion and displayed every reconstruction vertically flipped.


Milestone 1.2 investigates visual memory conditioning for canonical room-face
completion. Using local IP-Adapter conditioned on ranked source photographs
(selected via camera-to-face geometry, 3D point support, spatial proximity,
and semantic wall coverage), visual features and color palettes guide the
local inpainting process. Even when contextual evidence guides generation, all
generated pixels remain strictly IMAGINED in provenance (never relabeled as
RECONSTRUCTED or OBSERVED), protected pixels are restored exactly, and
context consistency checks (palette Delta E, structural edge continuity, and
ADE20K semantic unexpected additions) verify generation quality against
hallucinatory additions.

Milestone 1.3 turns the validated atlas + memory-conditioning pipeline into a
single, deterministic founder-presentable apartment-memory walkthrough. A
demo-visibility ranking (recovered-first, then intended walkthrough path
order, weighted by projected screen area) narrows missing faces down to only
those actually visible along the path, so generation effort is spent solely
on demo-relevant faces. A small fixed-seed candidate set is generated per
selected face and any candidate whose unexpected semantic additions exceed
tolerance, or that introduces unsupported signage/text on a structural wall,
is rejected outright and the face falls back to the deterministic engine
(Milestone 1.0) rather than accepting non-compliant output. Cross-face
harmonization corrects tonal seams between adjacent faces but is restricted
to pixels marked IMAGINED in provenance metadata; every other pixel is
asserted byte-exact before and after. A dedicated critical-region
demonstration proves that a protected class (art/poster/TV, detected via
ADE20K semantic segmentation from captured evidence when present) keeps its
exact captured pixels with only surrounding structure eligible for
generation, and that no person is fabricated. The presentation layer builds
one frozen 5-stop walkthrough (photo-origin view, a disclosed weak Remember
region, an explicit Remember-to-Imagine transition, the strongest immersive
Imagine viewpoint, and constrained movement clamped to the recovered room
envelope) and renders it in two modes from the same frozen assets: a
product mode that hides raw metrics, file paths, and filenames, and a
separate debug/provenance mode for internal review. All assets are
pre-generated; a deterministic backup recording (ffmpeg `xfade` over a fixed
frame sequence) is produced locally so the demo can run without live
inference if needed.

Milestone 1.3A resolves the unconstrained inpainting candidate rejections by
using deterministic structural Imagine output as initialization for missing
faces, followed by low-strength IP-Adapter memory-conditioned refinement
(`strength = 0.35–0.55`). The deterministic base anchors planar wall geometry
and edge boundaries, while low-strength refinement infuses photographic
lighting and palette from ranked source views without triggering semantic
hallucinations (doors, windows, sky, signage). All protected pixels remain
bit-exact, critical regions remain absent, and generated pixels maintain
strict IMAGINED provenance under the unchanged Milestone 1.3 semantic
consistency validator.

Milestone 1.4 adds a packaging boundary after validation. The freeze builder
has no reconstruction or generation entry point: it verifies hard-coded
SHA-256 identities for the accepted Remember atlas and the two accepted
Milestone 1.3A learned faces, verifies their recorded seed/strength and
protection results, and copies only those assets into an output directory
that must be outside Git. The resulting presentation is a self-contained
local HTML file with a restrictive Content Security Policy, a fixed five-stop
Remember/Imagine path, reset and optional provenance controls, and a
deterministic pre-rendered MP4. Startup performs no model loading, network
access, inference, reconstruction, or live generation.
