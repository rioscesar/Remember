# Representation spike

The point-cloud spike recovered valid geometry but failed the product test: the apartment was not recognizable as a place on a physical device. This spike tests whether source photographs, guided by recovered camera and geometry evidence, provide a better representation.

## Provenance vocabulary

| Class | Meaning | Allowed |
|---|---|---|
| Captured | Pixels directly present in a selected source photograph or video. | Yes |
| Reconstructed | Geometry or surfaces supported by multiple independent observations. | Yes |
| Interpolated | A view made by geometrically reprojecting or blending captured pixels between recovered viewpoints. | Yes, with explicit support and conflict masks |
| Imagined | Content generated without sufficient photographic evidence. | No |

Interpolation is not permission to hide uncertainty. Unsupported target pixels remain transparent, and mutually inconsistent observations are measured rather than blended away.

## Local prototype

`tools/representation_spike.py` creates a private, offline HTML comparison from an undistorted COLMAP workspace. It does not copy source media into Git. Its generated photographs, screenshots, metrics, and other derived artifacts belong in a local output directory outside the repository.

Install the pinned prototype dependencies:

```powershell
python -m pip install -r tools\requirements-representation.txt
```

Run the tool with paths to the local COLMAP text model, undistorted images, geometric depth maps, fused PLY, visibility sidecar, and `patch-match.cfg`. The output contains:

- an interactive spatial photograph graph;
- three depth-assisted midpoint reprojections;
- mutually consistent two-view masks;
- dominant-plane measurements;
- one bounded planar-proxy diagnostic;
- `metrics.json` with objective support and conflict percentages.

## Experiment A: depth-assisted source-pixel reprojection

For a virtual camera halfway between two recovered cameras, the prototype:

1. back-projects every geometrically valid depth pixel from each source view;
2. transforms that evidence into the virtual camera;
3. forward-projects the original source pixel without converting it to a point-cloud colour;
4. resolves occlusion using target depth;
5. blends only pixels whose two projected depths agree within 2%;
6. leaves all other target pixels transparent.

Three high-overlap camera pairs from the Clean reconstruction produced:

| Comparison | At least one source | Two sources | Mutually consistent | Conflict | Unsupported |
|---|---:|---:|---:|---:|---:|
| 1 | 15.28% | 3.29% | 2.47% | 0.82% | 84.72% |
| 2 | 15.09% | 3.42% | 2.77% | 0.64% | 84.91% |
| 3 | 12.36% | 2.50% | 2.11% | 0.39% | 87.64% |

The source pixels preserve more photographic detail than coloured points, but the representation is still fragments in empty space. Depth-assisted reprojection inherits the same coverage limit as dense MVS and does not materially improve recognition.

**Verdict: failed.**

## Experiment B: planar architectural proxy

A deterministic RANSAC pass searched the multi-view-supported fused cloud for dominant planes. The strongest candidate has:

- 38,913 supporting points;
- evidence from 9 recovered cameras;
- 4.98 by 1.48 reconstruction-unit robust extents;
- 0.0091 reconstruction-unit RMS residual under a 0.0452 threshold.

Four additional planes have 764 to 3,665 points and support from four or five cameras. Major planar geometry therefore is recoverable from the current evidence.

The prototype estimated a source-to-virtual-view homography from 20,323 observations of the strongest plane. Of these, 19,544 (96.17%) agree within a 2-pixel RANSAC threshold, with 0.64-pixel RMS reprojection error. Restricting the texture to the convex hull of those observations covers 12.10% of the target view and produces a clear photographic wall region.

That result is **not an acceptable representation**. The convex boundary spans foreground objects whose pixels are captured but whose geometry is not on the accepted wall plane. The diagnostic therefore flattens furniture and other occluders onto the wall. Restricting the mask to pixels with depth already verified to lie on the plane avoids that error but collapses coverage back toward the dense-depth result.

A production planar proxy needs separately supported architectural boundaries and conservative foreground occlusion masks. A plane fit alone is not evidence that every pixel inside its projected convex hull lies on that plane.

**Verdict: geometrically feasible, provenance boundary unresolved. Do not import into Android.**

## Experiment C: spatial photograph graph

Each registered photograph is an anchor at its recovered camera pose. Neighboring anchors are ranked by:

- shared reconstructed landmarks;
- recovered camera distance;
- viewing-direction difference.

Navigation moves only between recovered photograph positions. Original pixels remain visually dominant, direction and overlap determine the available moves, and every displayed image is 100% captured evidence.

For the Clean set this representation is immediately recognizable because it displays the apartment photographs themselves. It preserves walls, floors, furniture, and fine detail that neither points nor depth-assisted interpolation retained. It permits spatially meaningful discrete translation, but not continuous free-viewpoint movement.

**Verdict: useful navigation and fallback primitive. It has not been accepted as the primary product representation.**

## Comparison

| Representation | Recognizable | Preserves detail | Continuous translation | Evidence issue |
|---|---|---|---|---|
| Dense points | No | Poor | Yes | 97.4% depth absence under reliable filtering |
| Depth-assisted reprojection | No | Good where present | Limited | 84.7–87.6% unsupported |
| Planar proxy diagnostic | Partially | Good | Limited | Foreground is flattened without supported boundaries |
| Spatial photograph graph | Yes | Exact captured pixels | Discrete only | Navigation quality, not provenance |
| Nearest source photograph | Yes | Exact captured pixels | No | No spatial navigation |

## Architectural direction

Do not discard COLMAP. Its camera poses, correspondences, overlap graph, and depth estimates remain useful evidence. The supported direction is:

`photos -> camera/geometry evidence -> evidence-provenanced photographic surfaces and navigation`

Continuous interpolation may be layered between nearby anchors only where support metrics pass a future quality gate. It must not replace a captured photograph with a mostly unsupported virtual view.

## Stop/go result

Stop work on higher point density and unrestricted free-viewpoint rendering for this dataset. Do not modify the Android renderer yet. The next authorized experiment is semantic planar photographic reconstruction of the strongest recovered wall, with the spatial photograph graph preserved as a fallback rather than treated as the product conclusion.

The next experiment asks:

> Can source pixels classified as belonging to the recovered wall create a continuous photographic structural surface without requiring dense depth at every displayed pixel?

## Research references and licenses

- Levoy and Hanrahan, [Light Field Rendering](https://graphics.stanford.edu/papers/light/) (1996), and Gortler et al., [The Lumigraph](https://www.microsoft.com/en-us/research/publication/the-lumigraph/) (1996), establish image-based novel-view rendering but assume substantially denser ray sampling than this 11-camera set provides.
- Buehler et al., *Unstructured Lumigraph Rendering* (SIGGRAPH 2001), generalizes light-field rendering to irregular cameras with a geometric proxy. A practical open implementation is [manurare/ULR](https://github.com/manurare/ULR), licensed MIT. It is an algorithm reference only: its Triangle dependency is not licensed for unrestricted commercial inclusion.
- Debevec, Borshukov, and Yu, [Efficient View-Dependent Image-Based Rendering with Projective Texture-Mapping](https://www.pauldebevec.com/Research/VDTM/) (1998), supports choosing a small number of source photographs per proxy polygon. Remember explicitly rejects the paper's object-space hole-filling step.
- Snavely, Seitz, and Szeliski, [Photo Tourism: Exploring Photo Collections in 3D](https://www.microsoft.com/en-us/research/publication/photo-tourism-exploring-photo-collections-in-3d/) (2006), is the direct precedent for using recovered camera geometry to navigate original photographs.
- Furukawa et al., [Manhattan-world Stereo](https://www.microsoft.com/en-us/research/publication/manhattan-world-stereo/) (CVPR 2009), demonstrates plane hypotheses for calibrated, texture-poor architecture. Straub et al., [A Mixture of Manhattan Frames](https://openaccess.thecvf.com/content_cvpr_2014/html/Straub_A_Mixture_of_2014_CVPR_paper.html) (CVPR 2014), avoids forcing every surface into one global orthogonal frame.
- OpenCV documents [planar and rotation homographies](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html), [geometric image warps](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html), and [RANSAC feature-homography masks](https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html). OpenCV 4.5 and later is Apache-2.0.
- COLMAP is BSD-3-Clause; NumPy is BSD-3-Clause. Their bundled or transitive dependencies retain separate terms.

Paper publication pages are research references, not software licenses. Any production dependency still requires review of its exact version and transitive dependencies.
