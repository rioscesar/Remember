# Founder review — Milestone 1.4

Milestone 1.4 freezes the accepted Milestone 1.3A apartment-memory demo for presentation. It does not change reconstruction, models, generation, Android/AR, object completion, product scope, or any semantic threshold. The generalized builder (`tools/milestone14_demo_freeze.py`) can only verify and package pre-generated assets; it has no generation path.

## Frozen result

The private output contains exactly six files: one offline `index.html`, one sanitized freeze manifest, three accepted face PNGs, and one backup MP4. The builder rejects any output directory inside the Git working tree and deletes/recreates only the explicitly supplied private output directory.

| Face | Presentation role | Accepted configuration | SHA-256 |
|---|---|---|---|
| `front` | Remember view and real-painting critical-truth moment | Accepted Milestone 1.3 Remember atlas | `540218945c7d73a43b055f2d91fca2cf720b7fddf9326dc4aa7a7468d3d7e907` |
| `back` | Primary Imagine view | strength `0.35`, seed `1301` | `cb61cd732b6e40c7a794678edef5fd6f1bb7134f9bc92c1583bf102bef62abdd` |
| `right` | Constrained final Imagine view | strength `0.35`, seed `1101` | `d41b1f0591da46b8b0913d80756bd60cb0bb99e0ac62c4d319927a40ba7de01b` |

Both learned assets are byte-for-byte copies of the accepted private Milestone 1.3A candidates. Their accepted metrics are checked before copying: consistency passed, protected modifications `0`, and critical violations `0`. No candidate is regenerated.

## Presentation behavior

The default guided path is fixed at five stops and 36 seconds:

1. photo-origin Remember view;
2. critical-truth Remember view, explicitly identifying the real captured painting;
3. labeled Remember-to-Imagine boundary;
4. immersive Imagine view using the accepted back face;
5. constrained finish using the accepted right face.

The page provides Remember/Imagine controls, a restrained 900 ms visual transition, deterministic next/play controls, reset, keyboard advance/reset, responsive presentation styling, and an optional provenance panel. Rapid input is protected by a monotonically increasing render epoch so a stale transition cannot overwrite the most recent action. Reset and refresh both return to stop 1 in Remember mode.

Startup is a static local file load. A restrictive Content Security Policy permits only same-directory images and inline presentation code/styles; it permits no network connection. The bundle contains no model, model reference, inference call, reconstruction input, source-photo name, private path, analytics, or live-generation code.

## Measurements

| Check | Result |
|---|---|
| Cold desktop start, Edge headless, 1280×720 | `799.26 ms`, passed |
| Fresh narrow start / refresh behavior, 480×720 | `775.72 ms`, passed |
| Reset to default | passed |
| Rehearsal repeat / refresh | passed |
| Rapid Remember/Imagine toggles | passed; last action wins |
| Responsive resize | passed at both measured viewports |
| Guided path | 5 stops, `36.0 s` |
| Backup video | H.264, 1280×720, yuv420p, 24 fps, `36.000 s` |
| Backup SHA-256 | `810f7806203cb6ece840b1c5301d091c550df465615f6dc816e53c6cd6dc20b3` |
| Backup deterministic rebuild | passed; identical SHA-256 |

The browser checks launch the generated local page with background networking and component updates disabled. An internal presentation check performs rapid mode changes, navigates to the final stop, resets, and verifies the default state. A second fresh process exercises the narrow layout and refresh-default behavior.

## Truth, seams, and privacy

The critical-truth record is inherited from the accepted Milestone 1.3 evidence: ADE20K detected a real `painting` on the front face, detected-from-captured-evidence is true, protected modifications are `0`, critical violations are `0`, and no person was fabricated. The presentation uses the unchanged accepted front atlas rather than recreating a missing overlay or staging a synthetic replacement.

The two accepted Milestone 1.3 seam checks remain 2/2 passing. The freeze also measures the actual copied assets using mean CIE76 Delta E across five-pixel atlas borders:

| Frozen pair | Border mean ΔE | Review threshold | Result |
|---|---:|---:|---|
| front ↔ right | 15.30 | 38.0 | passed |
| back ↔ right | 14.03 | 38.0 | passed |

The private-directory audit found exactly the six expected files, zero unexpected files, zero network references, zero source-photo names, and zero private paths. The tracked-file audit found no image, video, model, or other private media extension in Git and no private capture/session identifiers.

## Validation

All private-data-free Python test programs passed:

- `face_guardrail_test.py`
- `learned_inpainting_test.py`
- `milestone09_synthetic_test.py`
- `milestone13_synthetic_test.py`
- `milestone13a_synthetic_test.py`
- `milestone14_synthetic_test.py`

The Milestone 1.4 suite covers hash rejection, outside-Git enforcement, exact accepted seed/strength provenance, real-painting truth requirements, five-stop/path-duration invariants, offline startup, product/debug controls, reset, resize behavior, and stale-transition resistance. Android tests were not applicable because no Android source or behavior changed; the local shell also had no configured Java runtime.

## Founder verdict

**A — frozen and presentation-ready.** The private demo and deterministic backup video are complete outside Git. The presentation uses only the exact accepted assets, exposes the Remember/Imagine truth boundary, includes the real-painting critical moment, starts without models/network/generation, and passes the measured resilience, seam, provenance, protected-pixel, critical-violation, and privacy gates. Work stops here for founder review; nothing was pushed.
