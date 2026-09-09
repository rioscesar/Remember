# Spatial-continuity photo transitions (Milestone 0.7)

Milestone 0.6 concluded B-PARTIAL: the spatial photo graph and support gate are valuable, captured photos are strong anchors, and radiance-field bridges are not good enough for this sparse legacy-photo set. Milestone 0.7 does not continue tuning radiance reconstruction. It asks a narrower question: **can transitions between two real, untouched captured photographs be made to feel like moving through the same room, without fabricating anything in between?**

No generative AI, inpainting, synthetic views, new capture, additional Gaussian training, or additional COLMAP tuning was used. Every method below starts on an untouched source photo, ends on an untouched source photo, and only ever displays pixels drawn from those two photos. The same 13-photo Clean set is unchanged; the same 11-camera graph from Milestone 0.6 is reused as-is.

Private photographs, camera poses, rendered frames, diagnostics, and the interactive prototype remain outside Git. Only the generalized selection/transition code and aggregate metrics are published.

## Phase 1: selected pairs

Pairs were chosen by a data-driven score (not chronological adjacency) that rewards many shared SfM landmarks, a meaningful-but-not-extreme angular viewpoint change, and non-trivial camera displacement -- implemented in `tools/spatial_transition_prototype.py::score_edge`, applied to the existing Milestone 0.6 graph.

| Pair | Shared landmarks | Camera displacement | Angular change | Feature-match inliers |
|---|---:|---:|---:|---:|
| A (poster wall / couch / TV, wide sweep) | 976 | 2.42 (SfM units) | 58.7° | 133/245 (54%) |
| B (couch / door / poster wall, forward move) | 565 | 6.95 | 39.2° | 252/367 (69%) |
| C (poster wall to far couch/door, wide sweep) | 766 | 2.27 | 66.0° | 109/209 (52%) |

All three pairs share the same recognizable structural anchors: the multi-poster gallery wall, the sectional couch, the TV, and the front door -- exactly the kind of stable objects the brief asked landmark continuity to be judged against. Pair B's geometry-consistency check (recovered 3D translation projected onto the source camera's forward vector) independently confirms it as a genuine forward move through the room, while pairs A and C are lateral/orbital moves around the same wall -- a useful mix of motion types for comparison.

## Phase 2: transition methods implemented

Four methods were rendered per pair (18 frames each for the interpolated ones):

- **Baseline A -- Hard cut**: Photo A, then instantly Photo B.
- **Baseline B -- Crossfade**: plain opacity blend, no geometry.
- **Candidate C -- Geometry-aligned camera move**: Photo A is progressively perspective-warped using the real feature-matched A→B homography (capped at 80% strength so A stays recognizably intact), simulating a camera move; regions warping outside the original frame fade to transparent (never filled in); the last 35% of the transition crossfades into the untouched Photo B.
- **Candidate D -- Correspondence warp**: Photo A is warped toward B using the same homography for the full transition, but every pixel's opacity is modulated by a support field built from the density of RANSAC-inlier feature matches near that pixel (a distance-transform of inlier locations, re-warped every frame so it tracks the moving image rather than staying anchored to A's original coordinates). Pixels with no nearby matched landmark fade toward the destination photo rather than being stretched. A credibility gate (≥25 inliers and ≥35% inlier ratio) was defined to force an honest crossfade fallback if a pair's match were too weak to trust -- all three selected pairs passed this gate (52-69% inlier ratios), so no fallback was triggered.

A separate "portal/parallax handoff" variant (Phase 2's Method C in the brief) was not built as a fifth distinct renderer: Candidate C above already combines geometry-driven perspective movement with a crossfade handoff before unsupported content dominates, which is functionally what that style asks for. Building a fourth near-duplicate renderer for 3 pairs would not have added new evidence.

## Phase 3: landmark continuity

Landmark-match diagnostics (`landmark_overlay.png` per pair, private) draw every RANSAC-inlier correspondence between the two source photos. For all three pairs, the gallery-wall posters, TV, and couch silhouette produce dense, geometrically consistent match lines (parallel, non-crossing) -- the shared landmarks move in a way consistent with the recovered viewpoint change, not randomly. No pair showed the "landmarks jump unpredictably" failure condition.

## Phase 4: support-aware fade behavior

- **Candidate C (geometry move)**: the fraction of the frame with no valid warped content (i.e., genuinely unsupported, faded to transparent) grows through the warp phase and then holds steady during handoff. It varies substantially by pair: Pair B (the forward move, highest inlier ratio) stays nearly fully supported throughout (0.7% unsupported at handoff), while Pairs A and C (the two wide lateral sweeps) lose roughly half the frame to transparency by handoff (48.7% and 55.9%). This is an honest, expected consequence of a single global homography: large-angle, low-forward-motion sweeps push more of the frame outside where a planar warp remains valid.
- **Candidate D (correspondence warp)**: mean per-pixel support (not fraction-unsupported, but a continuous 0-1 confidence) stayed in the 4-24% range across all three pairs and all frames -- i.e., even the "credible" pairs never claim strong support across most of the frame. This is the same honest conservatism seen in Milestones 0.5-0.6's support gate: it would rather under-claim than over-claim.

## Trust behavior / visual finding

Viewing the rendered frames directly surfaced an important, reportable artifact: **during the Candidate C handoff blend and throughout Candidate D, near-plane objects (the couch) visibly double/ghost against far-plane objects (the poster wall) that the single global homography cannot simultaneously align.** This is the textbook signature of real parallax that a 2D homography (a single-plane assumption) cannot resolve, and it is exactly the class of artifact the brief's failure conditions warned about ("warping creates rubbery/distorted imagery"). It is honest -- no content is invented, both layers are real pixels from real photos -- but it is visually distracting exactly where the room has meaningful depth variation (near couch vs. far wall).

The **first ~60% of Candidate C**, before the crossfade blend begins, does not show this problem: it only shows the warped Photo A alone, and reads as a genuinely convincing "the camera is panning/rotating across this recognizable wall" effect, especially for Pair A and Pair B. The problem is concentrated in the blend window and in Candidate D's full-duration blending.

## Comparison against baselines

| | Baseline A (hard cut) | Baseline B (crossfade) | Candidate C (geometry move) | Candidate D (correspondence warp) |
|---|---|---|---|---|
| Communicates spatial movement | No -- instant jump | Weak -- dissolve with no directional cue | Yes, during the pure-warp phase | Partially -- warp direction is right but blended too early/long |
| Landmark stability | N/A (no interpolation) | Landmarks visibly double throughout | Stable while unblended; doubles during handoff | Doubles for most of the duration |
| Distracting artifacts | None (but no continuity) | Mild ghosting | Present only in the final ~35% | Present throughout |
| Feels like "same room, new spot" | No | Marginal | Closest of the four, if trimmed to the pre-blend phase | Not currently |

**Best method: Candidate C (geometry-aligned camera move), but only its pre-blend phase.** Restricting Candidate C to roughly its first 60-65% (before the crossfade blend engages) and then cutting directly to Photo B, rather than blending through the parallax-mismatched region, would likely read as materially better than both baselines for Pairs A and B. As currently rendered with the full blend included, Candidate C is a clear improvement over Baseline B for the majority of its duration but re-introduces crossfade-like ghosting at the end.

## Remaining limitations

- A single global homography per pair cannot resolve real depth parallax (near couch vs. far wall); this is the direct cause of the ghosting artifact and is a fundamental limitation of 2D correspondence warping, not a tuning bug.
- Candidate D's full-duration blend suffers this problem for its entire length, not just a handoff window, making it currently the weaker of the two candidates despite passing the credibility gate.
- Results vary meaningfully by motion type: the forward-moving pair (B) tolerated the warp far better (0.7% unsupported at handoff) than the two wide lateral sweeps (A, C: ~50%+ unsupported) -- "good results only for one easy pair" is a real risk the brief warned about, and only Pair B currently looks strong across its full transition; Pairs A/C look strong only in the trimmed pre-blend phase.
- Only 3 of the graph's 33 edges were instrumented this milestone; the interactive prototype's "continue through the room" hooks are present (destination-photo neighbors are listed from the full graph) but non-instrumented edges are explicitly labeled as falling back to a plain crossfade rather than pretending every edge in the room has this treatment.

## Verdict: **B — PARTIAL**

Geometry-aligned camera movement over real photo pixels, cut before the crossfade blend engages, is a genuinely promising improvement over a hard cut or plain crossfade and is the strongest evidence yet in this project that spatial continuity between captured photos (not 3D reconstruction) is a viable product direction. However, the result is not yet consistently strong: it depends heavily on motion type (forward moves >> wide lateral sweeps), the current implementation's blend phase reintroduces the same ghosting problem it was meant to avoid, and Candidate D (correspondence warp) is not yet credible as a full-duration effect despite passing its numeric credibility gate. This does not clear the bar for "materially better than a hard cut or crossfade" across all three pairs as rendered today -- only for the pre-blend phase, and only reliably for the forward-motion pair.

## Founder review

The interactive prototype (`spatial-transitions-prototype-v1/index.html`, private/local) lets you compare all four methods per pair, scrub frame-by-frame, and view the landmark-match diagnostic overlay. Please judge for yourself whether the pre-blend Candidate C phase feels like "moving to another spot in the room" versus "being shown another picture," and whether trimming the blend window (a straightforward follow-up, not attempted here to avoid quietly relitigating the verdict) is worth pursuing before broader graph coverage. This report does not make that call for you.

Not proceeding to Android or AR. Stopping here for founder review per the milestone brief.
