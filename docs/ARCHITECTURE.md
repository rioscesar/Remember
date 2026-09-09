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
