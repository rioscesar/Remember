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
- [Development guide](docs/DEVELOPMENT.md)
- [Contributing](CONTRIBUTING.md)

The Milestone 0.9 private desktop walkthrough is built with
`tools/milestone09_walkthrough.py`. It renders a coarse six-face 3D room
shell in pure CSS (no WebGL/Three.js), with pointer-drag look and WASD/arrow
translation clamped to the recovered envelope, per-face evidence textures or
an explicit "no evidence" panel, and partial evidence-backed object cards
from residual (non-planar) point clusters. It emits only aggregate metrics
and provenance-coded views in the requested output directory; private
photos, poses, masks, and renders must remain outside the repository.

## Licensing

Copyright (c) 2026 Remember contributors. All rights reserved.

This repository is public for development visibility. No license is granted to
use, copy, modify, or distribute its contents.
