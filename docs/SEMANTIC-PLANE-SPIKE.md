# Semantic planar photographic reconstruction

This desktop-only experiment tests whether direct source pixels classified as belonging to a recovered structural wall can form a continuous photographic surface without requiring reconstructed depth at every displayed pixel.

It uses the existing private Clean reconstruction and does not change the Android application. Source photographs, semantic masks, renders, model weights, reconstruction artifacts, and identifying metadata remain outside Git.

## Method

1. Recover the strongest plane from fused points observed by at least three photographs.
2. Segment registered photographs locally with `openmmlab/upernet-convnext-tiny`, an ADE20K semantic segmentation checkpoint.
3. Select only semantic wall components connected to and inside the projected convex hull of the target plane's own observations. This bounds the semantic mask to the observed target wall rather than accepting every connected wall surface.
4. Validate masks against geometric depth:
   - recovered on-plane samples measure semantic wall recall;
   - any finite depth confidently off the target plane vetoes that pixel;
   - views with under 20% recall, over 15% geometric conflict, or under 55% agreement among decisive depth samples are rejected.
5. Estimate a plane homography from each accepted source into three nearby virtual views.
6. Warp original source pixels and the validated wall masks. Pixels outside the robust recovered plane extent remain transparent.
7. Weight sources by plane viewing angle, homography inliers, and reprojection RMS.
8. Bilinearly resample only pixels whose complete source interpolation footprint passed the wall mask. Blend only observations within 35 RGB levels of the highest-confidence source at that pixel. Conflicting observations do not contribute.
9. Emit a provenance raster alongside every result.

No inpainting, texture extension, monocular depth, synthesized pixel, generated geometry, or unsupported hole filling is used.

## Model and licensing

The final reported run uses [OpenMMLab UPerNet with ConvNeXt-Tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny), trained on ADE20K.

- Checkpoint metadata license: MIT.
- MMSegmentation: Apache-2.0.
- Transformers: Apache-2.0.
- PyTorch: BSD-style.
- OpenCV: Apache-2.0.

An initial diagnostic used NVIDIA SegFormer, but its upstream non-commercial license is not suitable for the project and none of its measurements are used below.

The model performs classification only. It does not generate or upload pixels.

## Geometry and mask validation

The target wall plane has 38,913 multi-view-supported points observed by nine cameras. Seven of nine useful source views passed the final semantic/geometry gate. Depending on the accepted source:

- semantic wall coverage was 25.5–40.2% of the photograph;
- recall inside the projected target-plane support hull was 22.5–53.3%;
- 56.3–76.7% of decisive depth samples agreed with the target plane;
- semantic target-wall pixels contradicted by off-plane geometry were 2.9–10.2%.

Low recall is conservative: it removes wall evidence. Foreground conflict is the safety-critical quantity because it indicates non-wall pixels that would otherwise be flattened onto the wall.

## Novel-view results

Three virtual cameras were rendered at 25%, 50%, and 75% between two nearby recovered camera poses:

| Measurement | View 1 | View 2 | View 3 |
|---|---:|---:|---:|
| Recovered wall footprint covered | 78.50% | 77.23% | 77.44% |
| Full-frame direct source pixels | 9.97% | 9.32% | 8.89% |
| Full-frame multi-view captured | 8.72% | 8.14% | 7.78% |
| Full-frame single-view captured | 1.25% | 1.18% | 1.11% |
| Unsupported full frame | 90.03% | 90.68% | 91.11% |
| Mean non-reference overlap RGB disagreement | 24.37 | 24.97 | 25.21 |
| Full-frame high photometric conflict | 4.28% | 4.38% | 4.50% |
| Supported pixels with at least one conflicting source | 42.89% | 47.03% | 50.62% |
| Internal boundary pixels with conflict | 8.42% | 8.05% | 9.43% |

The full-frame support percentage is low because this experiment intentionally renders one wall only. Within the recovered wall footprint, coverage is approximately 77–79%. These are deliberately easy nearby novel views: 25%, 50%, and 75% between the closest accepted camera pair, separated by 0.164 reconstruction units and 8.2 degrees. They establish local interpolation, not unrestricted movement.

## Provenance

The prototype reserves these representation values:

| Value | Class | Use in this experiment |
|---:|---|---|
| 0 | `UNSUPPORTED` | Transparent output |
| 1 | `CAPTURED_SINGLE_VIEW` | One validated source photograph |
| 2 | `CAPTURED_MULTI_VIEW` | Two or more source photographs consistent with the highest-confidence source |
| 3 | `GEOMETRICALLY_INFERRED` | Reserved for geometry, not emitted as visible pixels |
| 4 | `GENERATED` | Prohibited; emitted count is zero |

Every visible output pixel is bilinearly resampled and, where consistent with the highest-confidence observation, weighted from one or more source photographs. The wall plane determines placement but does not manufacture appearance. Generated pixels are prohibited by construction and remain zero.

## Visual verdict

The result materially differs from point rendering: the wall reads as one continuous photographic surface from nearby novel viewpoints, rather than isolated artwork and edge landmarks floating in black. The spatial arrangement of the wall and its attached decoration remains recognizable.

The result is not production-ready:

- foreground and wall-art exclusions create conspicuous holes;
- the 512-pixel semantic model misses or partially labels small artwork;
- residual exposure and warp differences remain visible;
- 43–51% of supported pixels reject at least one conflicting source observation, although those conflicting samples do not contribute to the output;
- approximately 8–9% of internal mask-boundary pixels coincide with photometric conflict;
- this experiment establishes one wall only, not a room.

**Verdict: the core semantic-planar hypothesis passes for the test wall, but boundary quality remains unresolved.** Per the founder stop condition, floor and ceiling are not being promoted to product work and Android remains unchanged. Founder review is required before either refining wall-attached-object handling or repeating the method for additional structural surfaces.

## Reproduction

Install the pinned desktop dependencies:

```powershell
python -m pip install -r tools\requirements-representation.txt
```

Run `tools/semantic_plane_spike.py` with the local COLMAP text model, undistorted source images, depth maps, fused PLY, visibility sidecar, and `patch-match.cfg`. The output directory contains private masks, three RGBA wall views, provenance rasters, `metrics.json`, and a local `index.html`; it must remain outside Git.
