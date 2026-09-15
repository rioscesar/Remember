# Remember

Remember helps people spatially revisit places held in their existing photographs. It reconstructs only what the photographs can support.

## Getting started

1. Open this folder in the latest stable version of Android Studio.
2. Allow Gradle to finish syncing the project.
3. Select an emulator or connected Android device and run the `app` configuration.

The project uses Kotlin and AndroidX. It targets Android 15 (API 35) and supports Android 7.0 (API 24) and later. The private local Companion currently performs sparse reconstruction; the Android app imports and explores its evidence-backed result.

## Documentation

- [Product principles](docs/PRODUCT-PRINCIPLES.md)
- [Reconstruction feasibility spike](docs/RECONSTRUCTION-SPIKE.md)
- [Representation pivot spike](docs/REPRESENTATION-SPIKE.md)
- [Semantic planar reconstruction spike](docs/SEMANTIC-PLANE-SPIKE.md)
- [Photographic coherence spike](docs/WALL-COHERENCE-SPIKE.md)
- [Radiance-field representation spike](docs/RECONSTRUCTION-REPRESENTATION-SPIKE.md)
- [Sparse-view regularization and support-gate spike](docs/SPARSE-VIEW-SUPPORT-GATE-SPIKE.md)
- [Hybrid supported spatial memory prototype](docs/HYBRID-SPATIAL-MEMORY-SPIKE.md)
- [Spatial-continuity photo transitions](docs/SPATIAL-TRANSITION-SPIKE.md)
- [Evidence-maximizing structural completion](docs/EVIDENCE-MAXIMIZING-RECONSTRUCTION-SPIKE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Privacy](docs/PRIVACY.md)
- [Milestone 0 results](docs/MILESTONE-0-RESULTS.md)
- [Milestone 0.9 founder steering report](docs/MILESTONE-0.9-FOUNDER-REPORT.md)
- [Milestone 1.0 founder steering report](docs/MILESTONE-1.0-FOUNDER-REPORT.md)
- [Milestone 1.1 founder steering report](docs/MILESTONE-1.1-FOUNDER-REPORT.md)
- [Milestone 1.2 founder steering report](docs/MILESTONE-1.2-FOUNDER-REPORT.md)
- [Milestone 1.3 founder steering report](docs/MILESTONE-1.3-FOUNDER-REPORT.md)
- [Development guide](docs/DEVELOPMENT.md)
- [Contributing](CONTRIBUTING.md)

The Milestone 1.1 private desktop walkthrough is built with
`tools/milestone09_walkthrough.py`. It renders a coarse six-face 3D room
shell in pure CSS (no WebGL/Three.js), with pointer-drag look and WASD/arrow
translation clamped to the recovered envelope. The walkthrough has a clear
**Remember** view for evidence-backed OBSERVED / RECONSTRUCTED / INFERRED
faces, an **Imagine** view that first attempts validated local learned
inpainting on the strongest missing canonical face and otherwise falls back
truthfully to the Milestone 1.0 deterministic IMAGINED structural atlas
fallback, and a debug provenance view. Completion happens in canonical
room-face/atlas space, never frame-by-frame; OBSERVED/RECONSTRUCTED,
non-generatable, and critical regions are restored/locked exactly,
ambiguous/critical unknowns remain ABSENT, and object gap completion is
disabled unless confidence is adequate. The learned path is local-only,
fixed-seed, fail-closed, and only becomes the default when the configured
Diffusers inpainting model is available and validates. The tool emits
aggregate metrics and provenance-coded views in the requested output
directory; private photos, poses, masks, models, and renders must remain
outside the repository.

## Licensing

Copyright (c) 2026 Remember contributors. All rights reserved.

This repository is public for development visibility. No license is granted to
use, copy, modify, or distribute its contents.
