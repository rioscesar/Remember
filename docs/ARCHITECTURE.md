# Architecture

Remember separates private reconstruction from private viewing.

## Components

| Component | Responsibility | Media boundary |
|---|---|---|
| Android app | Memory library, source URI references, bundle validation, and sparse-scene rendering | Reads selected media through Android's document provider; has no `INTERNET` permission. |
| Remember Companion | Local CPU Structure-from-Motion with COLMAP and bundle export | Runs on the person's computer; input and temporary output stay local. |
| Reconstruction bundle | Versioned JSON carrying points, quality values, and source-photo names | Imported manually from local storage; contains no generated scene completion. |

## Android data flow

1. The person names a memory and chooses at least three photos through the system document picker.
2. The app retains read-only URI permissions and display names; it does not copy or alter the originals.
3. The Companion reconstructs a sparse point cloud from matching evidence and exports `reconstruction.json`.
4. The app verifies the bundle format, minimum evidence, and connection to the selected photo names before persisting it.
5. The scene view projects the imported three-dimensional landmarks in response to touch rotation. Point opacity decreases with observed support and reprojection quality.

No distance is displayed: ordinary-photo SfM has arbitrary scale unless separate measured evidence is introduced.

## Evidence contract

An accepted landmark has coordinates, observed RGB color, at least three supporting photographs, and a non-negative reprojection error. Bundles also carry aggregate pre- and post-filter landmark counts. The app rejects bundles with fewer than 100 accepted landmarks. It renders no surfaces, inferred depths, or generated regions. The background and attenuated points form the evidence boundary rather than a fabricated room.

## Future boundaries

On-device SfM, dense geometry, and AR are deliberately separate follow-on spikes. They cannot replace the provenance checks in the import contract.
