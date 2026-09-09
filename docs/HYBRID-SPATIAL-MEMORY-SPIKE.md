# Hybrid supported spatial memory prototype (Milestone 0.6)

This desktop-only Milestone 0.6 experiment asks whether constraining navigation to evidence-supported photographic space -- rather than trying to synthesize arbitrary free viewpoints -- produces a more compelling and trustworthy sense of "moving through one remembered place" than ordinary photo browsing. It does not change the Android application, uses no generative content, and does not broaden the 13-photo Clean set (11 of 13 are registered and used here, unchanged from Milestones 0.4/0.5).

Private photographs, camera poses, rendered bridge frames, and the interactive prototype remain outside Git. Only the generalized graph/rendering/prototype-assembly code and aggregate metrics are published.

## Phase 1-2: spatial viewpoint graph

Built a graph over the 11 registered cameras. Edges require both shared SfM landmarks (≥8) and a camera-geometry-only "support-gate continuity" score sampled along the straight path between camera centers (same opacity-independent gate validated in Milestone 0.5), evaluated against every *other* registered camera, not just the two endpoints.

| Metric | Value |
|---|---:|
| Registered camera nodes | 11 |
| Connected nodes | 11 (0 isolated) |
| Total edges | 33 |
| Strong edges (local radiance bridge credible) | 1 |
| Weak edges (partial fade / photo handoff only) | 32 |
| Average shared landmarks per edge | 260 |
| Average min-path support, strong edges | 0.541 |
| Average min-path support, weak edges | 0.359 |

Every node ended up connected to at least one other node, but the overwhelming majority of relationships (32/33) are only weak bridges. Only one pair of photographs (`20260709_192425.jpg` <-> `20260709_192437.jpg`) has evidence strong enough along its entire path to justify a local radiance bridge. This is an honest, expected consequence of an 11-photo walkthrough: most photo pairs are spatially related but not densely enough re-observed in between to support novel-view rendering.

## Phase 3-5: anchor experience, local movement, transition design

- **Anchors** always show the real captured photograph at full resolution/dominance -- the renderer is never used to reproduce a pose a real photo already covers. This was validated as the right call: even evaluating the regularized Milestone 0.5 checkpoint at the exact pose of a real training camera, the per-pixel support gate (which is driven by the model's own rendered depth, not opacity) only labeled 24-34% of pixels "supported" at those endpoint poses, because the underlying radiance model's depth channel is still noisy in artifact-heavy regions. A real photograph has no such uncertainty, so anchors must stay photographic.
- **Strong edge** (1 of 33): implemented as a 9-frame local radiance cross-dissolve using the Milestone 0.5 regularized Gaussians, with every frame's alpha faded per-pixel by the camera-geometry support gate (never by renderer opacity). Mean support across the bridge's frames ranged 0.32-0.48, and unsupported-pixel share ranged 1.4%-11.7% depending on position along the path -- the middle of the bridge is not meaningfully more supported than the ends, so the "bridge" is honestly patchy rather than a clean corridor.
- **Weak edges** (32 of 33): deliberately do **not** attempt reconstruction. The prototype crossfades directly between the two real captured photographs (Option 1's spirit, but with zero synthesized pixels) and the UI frames this explicitly as a "hand off to the next captured photo," per the founder's Option-2-style handoff guidance.
- Portal/window handoff (Option 2) and correspondence-based image morphing (Option 3) were not built as separate renderers this milestone -- with only one strong edge and 32 weak ones, the highest-value comparison was strong-bridge-vs-weak-crossfade, not a wider transition-style sweep. This is a scope reduction the founder should be aware of, not an oversight: building 2-3 additional transition styles for 32 structurally identical weak edges would not have added new evidence.

## Phase 6: desktop interactive prototype

Built a single local HTML/JS prototype (`hybrid-prototype-v1/index.html`, private) with three switchable modes over the same 11-photo graph:

- **A -- Chronological**: plain next/previous through all 11 photos in capture order.
- **B -- Spatial graph (hard cut)**: click directly to any spatially connected neighbor; no transition, immediate cut.
- **C -- Hybrid supported**: anchors dominate; the one strong edge plays the support-gated radiance cross-dissolve; all weak edges crossfade directly between real photos; a 2D graph panel shows the current node, its neighbors, and edge classification live.

## Phase 7: comparison

This is a **founder-judgment task by design** -- the brief explicitly asks not to answer the perceptual questions on the founder's behalf. The prototype is ready for side-by-side A/B/C comparison; the founder's assessment of recognition, spatial understanding, presence, continuity, trust, memory, and emotional threshold is the actual Phase 7 output and is not pre-empted here.

## Support integrity

Consistent with Milestone 0.5: at no point does the prototype trust renderer opacity to decide what is shown. The single radiance bridge is gated by the same independent camera-geometry support model, and every frame the founder sees during that bridge reports its own supported/weak/unsupported percentages live in the UI. Weak edges never invoke the radiance model at all, so there is zero risk of a confidently-wrong render appearing on 32 of the 33 relationships in this graph.

## Objective metrics summary

| Metric | Value |
|---|---:|
| Connected camera nodes | 11 / 11 |
| Graph edges | 33 (1 strong, 32 weak, 0 disconnected pairs among evidenced pairs) |
| Average shared landmarks per edge | 260 |
| Strong-bridge frames rendered | 9 |
| Strong-bridge mean support range across frames | 0.32 - 0.48 |
| Strong-bridge unsupported-pixel range across frames | 1.4% - 11.7% |
| Moments requiring hard handoff to a source photo | 32 of 33 relationships (all weak edges) |
| Floaters/artifacts encountered in the bridge | Same needle/streak artifact family documented in Milestones 0.4/0.5; visibly present but substantially masked by per-pixel support fading in low-confidence areas |

## Provenance

- Anchors: **CAPTURED** (original photographs, undegraded).
- Strong-edge bridge frames: **RECONSTRUCTED**, gated by an evidence-only support model; never blended with unsupported/opaque-but-wrong pixels.
- Weak-edge transitions: **CAPTURED**-to-**CAPTURED** crossfade only; no reconstruction involved.
- **INFERRED**: none used. **IMAGINED**: none used, anywhere in this milestone.

## Remaining limitations

- Only 1 of 33 evidenced relationships supports a radiance bridge at all; the hybrid experience is therefore mostly "connected photo browsing with one bridged exception," not yet a broadly bridged spatial memory.
- The strong bridge itself is not uniformly supported along its length (down to ~32% mean support mid-path), so even the best case is a patchy, fading bridge rather than a solid corridor.
- The support gate's own signal is noisier at real camera poses than expected (24-34% "supported" at true training poses) because it depends on the radiance model's rendered depth, which is still artifact-prone -- this reinforces keeping anchors purely photographic, but means the gate should be read as a conservative/lower-bound confidence signal, not a precise one.
- Weak-edge crossfades are functionally very close to Baseline B navigation; whether this materially exceeds Baseline A (chronological) or Baseline B (spatial hard-cut) in felt presence is exactly the open founder-review question this milestone was built to surface, not something the numbers alone resolve.

## Verdict: **B -- PARTIAL**

The spatial graph, support gate, anchor-first design, and support-gated radiance bridge all worked as engineered and are ready for founder evaluation. However, with only one evidence-qualified strong bridge out of 33 relationships in this 11-photo set, the hybrid experience is not yet demonstrated to "materially exceed chronological photo browsing" at scale -- that determination requires the founder's own comparison across modes A/B/C in the delivered prototype, which is the explicit next step, not a technical conclusion this report can make on the founder's behalf.

## Founder review

The interactive prototype (`hybrid-prototype-v1/index.html`, private/local) is ready. Please compare modes A, B, and C directly and answer the seven founder-review questions from the Milestone 0.6 brief yourself; this report does not answer them.
