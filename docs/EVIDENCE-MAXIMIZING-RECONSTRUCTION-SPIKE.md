# Evidence-maximizing structural completion (Milestone 0.8)

Milestone 0.7 concluded B-PARTIAL: spatial-continuity transitions between real photographs are promising but still felt like "being shown pictures," not "moving through a room," and the reason is structural, not stylistic -- Milestones 0.5-0.7 all treated *any* unsupported pixel as equivalent to *no photographic evidence*, so recognizable surfaces that were genuinely photographed (just not densely enough for per-pixel depth) rendered as black voids or isolated floating landmarks.

The founder revised the product philosophy for Milestone 0.8: stop requiring "no inference anywhere." Use as much real photographic evidence as possible, reconstruct aggressively from multiple views, **conservatively infer low-risk missing structure**, and **never confidently invent identity-critical content**. This is **Representation Doctrine v2**:

| Tier | Meaning | May this milestone infer it? |
|---|---|---|
| OBSERVED | a single captured photo directly supports this pixel | n/a -- always kept |
| RECONSTRUCTED | two or more captured photos agree on this pixel | n/a -- always kept |
| INFERRED | no direct capture; conservatively completed from nearby evidence, bounded distance, no learned/generative prior | only for structural, low-risk content (wall/floor/ceiling continuation) |
| IMAGINED | invented faces, artwork, text, objects, poses | **never produced by this codebase** (reserved code, permanently unused) |

Critical content -- people, faces, pets, handwriting, text/signage, artwork, TV screens, mirrors, distinctive personal objects -- must **never** be inferred over, at any gap size. Structural surfaces (wall/floor/ceiling continuations, corners, generic furniture backsides) may be conservatively completed, purely for spatial presence, not storytelling.

No generative AI, inpainting-as-hallucination, synthetic views, new capture, additional Gaussian training, additional COLMAP tuning, Android, AR, or cloud processing was used. The same 13-photo Clean set (11 registered views) is unchanged. All private photos, masks, atlases, and prototype renders remain outside Git; only generalized code, methodology, and aggregate metrics are committed.

## Doctrine v2 implementation

`tools/evidence_doctrine.py` is the single shared module every other script imports -- it is the *only* place in the codebase allowed to fabricate a pixel:

- Provenance codes `ABSENT=0, OBSERVED=1, RECONSTRUCTED=2, INFERRED=3, IMAGINED=4` (the last reserved and never assigned).
- ADE20K-150 label -> risk-tier classification (`STRUCTURAL_LABELS` / `CRITICAL_LABELS`, default `object`). Critical labels include `person, painting, picture, poster, animal, signboard, sign, bulletin board, clock, flag, sculpture, mirror, television receiver/television/tv/screen/crt screen` -- TV was added after the first segmentation pass showed it defaulting to the medium-risk `object` tier, which under-protects a screen the same way the brief explicitly warned against.
- `bounded_inference_fill()`: classical (non-learned) `cv2.inpaint` (Telea) restricted to unsupported, non-critical pixels within a capped distance of the nearest real evidence. Pixels beyond that distance, or ever flagged critical, are left `ABSENT` unconditionally. Returns a per-pixel provenance delta so every inferred pixel is auditable.

## Phase 2: scene risk segmentation

`tools/scene_risk_segmentation.py` re-uses the same locally cached `openmmlab/upernet-convnext-tiny` ADE20K model validated in Milestones 0.2-0.3 (fully offline, `HF_HUB_OFFLINE=1`) to classify every pixel of all 11 registered photos:

| Tier | Mean % of frame |
|---|---:|
| Structural | 65.0% |
| Object | 31.5% |
| Critical | 3.5% |

Visual spot-checks of the overlay confirm the classification is sound: walls/floor/ceiling green (structural), couch/cabinets/table blue (object), posters, the framed sketch, and the TV screen red (critical).

## Phases 3-5: wall structural completion

`tools/structural_completion.py` rebuilds the strongest recovered wall plane (9 views, 38,913 multi-view points -- the same plane validated in the Milestone 0.3-era wall-coherence work) via the *unmodified* `wall_coherence_spike.py` atlas pipeline (`detect_planes` -> `fit_plane` -> per-view homography -> `build_atlas`), then adds two new steps:

1. Each accepted view's Phase-2 critical mask is warped through that view's own homography into atlas space and OR-accumulated -- any atlas texel any view ever called critical is permanently excluded from inference, even where the atlas otherwise has partial support.
2. Remaining unsupported, non-critical texels are passed through `bounded_inference_fill` (max fill distance 140px in a 2400px-wide atlas).

| Metric | Value |
|---|---:|
| Accepted source views | 7 of 11 |
| Wall atlas OBSERVED+RECONSTRUCTED coverage | 79.9% (matches the prior validated wall-atlas result exactly, confirming the reused pipeline is unmodified) |
| Wall atlas critical-protected area | 37.7% (the gallery-wall poster cluster and TV are large relative to this wall) |
| Wall atlas newly INFERRED | 2.4% |
| Wall atlas still ABSENT | 17.8% |
| **Critical pixels ever inferred over** | **0** |

Visual inspection (`wall-atlas-completed.png`) confirms the intended behavior directly: gaps between posters are seamlessly, subtly filled with plausible wall texture, while every poster, the framed sketch, and the TV remain solid black holes -- honestly absent rather than completed.

### Floor and ceiling: a deliberate scope decision

Floor and ceiling are **not** ray-traced from geometry. The recovered point cloud is **not gravity-aligned** (`detect_planes` returns no axis-aligned normal), and multi-view point support for floor/ceiling planes is 4-5 views / 764-3,665 points -- roughly an order of magnitude weaker than the wall's 9 views / 38,913 points. Attempting precise per-pixel floor/ceiling reconstruction from that little evidence risked exactly the "confidently wrong" failure mode Doctrine v2 exists to prevent.

Instead, floor and ceiling are built as simple vertical-gradient colour bands, anchored to the real mean colour observed at the wall atlas's own top/bottom edges, and are **entirely and honestly labeled INFERRED** -- no claim of reconstructed geometry is made. This is reported, not hidden:

| | Value |
|---|---:|
| Combined floor+ceiling share of the stitched room image | 47.6% |
| Whole room image: OBSERVED+RECONSTRUCTED | 41.9% |
| Whole room image: INFERRED | 48.8% |
| Whole room image: ABSENT | 9.3% |

Nearly half the presented room image is honestly low-fidelity structural presence, not photographic evidence. This is the single most important number in this report and should weigh heavily on the verdict below.

## Phases 6-7: continuous room prototype

`tools/build_room_prototype.py` produces a small pannable HTML viewer (`room-prototype-v1/index.html`, private) over the stitched ceiling+wall+floor image, with one toggle:

- **Founder view** (`room-structural.png`): natural colours; the wall's inferred gaps blend in via real texture continuation; floor/ceiling bands are deliberately flat and desaturated so they read as "presence" rather than pretending to be detail; critical content is plain darkness.
- **Debug provenance view** (`room-provenance.png`): flat colour-coded per-pixel provenance (white=OBSERVED, pale blue=RECONSTRUCTED, muted violet=INFERRED, near-black=ABSENT/critical-protected). No loud debug colours appear in the founder view.

This replaces the black-void/isolated-landmark feel of Milestones 0.6-0.7 with one continuous horizontal strip the founder can drag through -- the smallest true test of "continuous structure" available without new capture or reconstruction work.

## Phase 8: ordinary object completion (couch) -- honest negative result

`tools/object_completion_test.py` deliberately does **not** attempt 3D object-mesh completion (out of scope this milestone). It measures a narrower, honest proxy: does a 2D homography-warped union of the couch mask from the two strongest couch-visible photos increase evidenced coverage versus the best single view?

| Metric | Value |
|---|---:|
| Best single-view couch coverage (of destination frame) | 21.7% |
| Combined two-view coverage (proxy) | 14.3% |
| Newly evidenced by the second photo | 1.0% |

The combined estimate is *lower* than the single best view, and the visual overlay (`couch-combined-coverage.png`) shows exactly why: the warped mask from the other photo lands mostly on the TV, not the couch, because a single planar homography cannot represent a non-planar couch viewed from a meaningfully different angle. This is a genuine, useful negative result: it confirms, with evidence rather than assumption, that the founder's decision to scope object handling to classification-only (not shape completion) this milestone was correct -- a naive 2D approach actively produces wrong answers here.

## Phase 9: human-face guardrail test

`tools/face_guardrail_test.py` uses a **purely synthetic, procedurally drawn** stand-in image (OpenCV ellipse/line primitives -- no real photo, no generated imagery) with a "face" region marked simultaneously unsupported and critical, surrounded by ordinary unsupported-but-non-critical "wall." It calls the real production `bounded_inference_fill` function directly (not a reimplementation).

| Metric | Value |
|---|---:|
| Critical region pixels | 28,281 |
| Critical pixels ever inferred over | **0** |
| Critical pixels remaining honestly ABSENT | 28,281 (100%) |
| Non-critical unsupported pixels healed | 16,990 |

The rendered result (`synthetic-after-fill.png`) shows the surrounding wall/floor seamlessly healed while the face region is a clean, unmodified black void -- proving the guardrail holds at the code level, not just by convention.

## Structural inference (summary)

Conservative inference worked exactly as designed for the wall (2.4% of atlas newly filled, 0 critical violations) and was used transparently, not silently, for floor/ceiling (48.8% of the presented room, explicitly labeled and reasoned about above). No inference ever extrapolated beyond its capped distance or touched a critical pixel, in both the real 11-photo dataset and the synthetic guardrail test.

## Evidence usage

Real photographic evidence dominates the wall (79.9% observed/reconstructed) but only a minority of the full presented room image (41.9%), because floor/ceiling evidence in this specific dataset is too weak to reconstruct responsibly. The wall result is a genuine improvement over Milestone 0.6/0.7's all-or-nothing treatment of unsupported pixels; the whole-room result is not yet "evidence-dominant" end to end.

## Critical details protection

Zero critical-region violations across both the real 11-photo wall atlas (37.7% of that atlas was critical-protected) and the synthetic guardrail test. This is the strongest, most unambiguous result in this milestone.

## Object completion

Classification-only, as scoped. The one completion experiment attempted (2D homography couch-coverage proxy) produced a clear, informative negative result rather than a false positive -- it did not need rescuing, and none was applied.

## Presence

The continuous room prototype communicates "one place" instead of isolated photo islands, but nearly half of that presence is deliberately low-fidelity inferred colour, not photographic detail. Whether that trade is acceptable for the product is a founder call, not a technical one.

## Trust

Every output pixel carries an explicit, auditable provenance code. The debug provenance view exists specifically so the founder (or, later, an in-product mode) can verify this claim rather than trust it blindly.

## Provenance

`room-provenance-codes.npz` (private) stores the full per-pixel provenance array for the stitched room image; `structural-completion-metrics.json` and `face-guardrail-metrics.json` (both reproducible from committed code, not committed themselves since they were generated from private imagery in this run) report the aggregate breakdown reflected in the tables above.

## Verdict: **B — PARTIAL**

The doctrine itself succeeded without qualification: conservative structural inference and absolute critical-content protection both worked, with measured zero violations on real data and on a synthetic guardrail test built specifically to catch a violation. The wall completion and the continuous room prototype are genuine, demonstrable improvements over Milestones 0.6-0.7's black-void/isolated-landmark experience.

This is not yet A-SUCCESS because the whole-room result still depends more on honest low-fidelity inference (48.8%) than on photographic evidence (41.9%), and the one object-completion experiment attempted this milestone confirmed that a straightforward 2D approach is not viable for non-planar furniture -- both are real, unresolved gaps, not just caveats. It is well clear of C-FAILED: the core doctrine risk (fabricating identity-critical content) was tested directly and never happened.

## Founder review

Please look at `room-prototype-v1/index.html` (private, drag to pan, toggle founder/debug views) and judge for yourself:

- Does the wall's structural completion read as genuine presence, or does 2.4%-of-atlas inference feel unnecessary given 79.9% is already real?
- Is a 48.8%-inferred floor/ceiling an acceptable trade for "the room feels continuous," or does it need to be visually distinguished further (or removed) before this direction continues?
- Does critical-content protection (posters/TV as solid black gaps) feel honest, or does it need a softer visual treatment before a founder-facing product would use it?
- Is the couch-completion negative result reason enough to leave object completion entirely out of scope for now, or worth a different (non-homography) approach later?

This report does not answer those on the founder's behalf. Not proceeding to Android, AR, or generative Imagine mode. Stopping here for founder review per the milestone brief.
