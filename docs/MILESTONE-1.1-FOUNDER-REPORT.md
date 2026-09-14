# Founder steering update — Milestone 1.1

Milestone 1.1 adds an optional local learned-inpainting engine to the existing
Milestone 1.0 Remember/Imagine walkthrough without changing reconstruction,
Android, AR, or the deterministic fallback. The new path is generalized around
canonical room-face atlases: it receives explicit GENERATABLE, LOCKED, and
CRITICAL masks; uses a fixed seed and plain structural prompts; restores every
protected pixel exactly after generation; accepts output only when protected
modification count is `0` and critical violations are `0`; and labels accepted
generated pixels **IMAGINED**.

## What changed

- Added `tools/learned_inpainting.py`, a local-only Diffusers inpainting module
  targeting `stable-diffusion-v1-5/stable-diffusion-inpainting` first.
- Added `tools/requirements-learned-inpainting.txt` for an isolated
  learned-inpainting package install that reuses the existing CUDA Torch rather
  than replacing reconstruction dependencies.
- Updated `tools/milestone09_walkthrough.py` with `--imagine-engine
  auto|deterministic|learned`. `auto` tries learned generation on the strongest
  missing face first, then scales to remaining faces only if that first face
  validates. If it does not validate, the existing deterministic structural
  fallback remains the active Imagine path.
- Added `tools/learned_inpainting_test.py`, a synthetic no-private-media test
  covering exact protected-pixel restoration, critical-mask enforcement,
  IMAGINED provenance assignment, structural prompt shape, and fail-closed
  missing-model behavior.

## Environment and model status

An isolated package directory was created outside the repository:

```text
C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\learned-inpainting-packages
```

It was installed with compatible local packages:

```text
diffusers==0.30.3
transformers==4.44.2
accelerate==0.33.0
safetensors==0.4.5
```

The run used the existing CUDA Torch interpreter from the separate artifact
environment:

```text
Python 3.10.11
torch 2.1.2+cu118
CUDA available: true
```

No private models, photos, renders, masks, or metadata were added to Git.

## Founder-review run result

The learned engine was requested in `auto` mode. It selected the strongest
missing face first: `back`. Dependencies were present, but the target model was
not available in the local Hugging Face/Diffusers cache. Outgoing model lookups
were deliberately disabled by default (`local_files_only=True`), and no cloud
processing was used. The learned path therefore failed closed and reported the
blocker instead of pretending generation happened:

```text
unable to load stable-diffusion-v1-5/stable-diffusion-inpainting from local cache:
Cannot find an appropriate cached snapshot folder for the specified revision on
the local disk and outgoing traffic has been disabled.
```

Because the first learned face did not validate, no remaining faces were sent to
learned generation. The walkthrough used the deterministic Milestone 1.0
structural atlas fallback for missing faces, with all generated fallback pixels
still marked IMAGINED.

## Measured metrics

| Metric | Result |
|---|---:|
| Remember faces with evidence | 3 / 6 |
| Missing faces completed by fallback | 3 |
| First learned target face | back |
| Learned first-face accepted | false |
| Learned model | stable-diffusion-v1-5/stable-diffusion-inpainting |
| Learned target resolution | 480 x 288 |
| Learned steps / guidance / seed | 24 / 6.0 / 1101 |
| Learned attempt latency | 2581.79 ms |
| Walkthrough generation latency | 4156.60 ms |
| VRAM before / after | 243 MB / 243 MB |
| VRAM delta | 0 MB |
| Generated object gaps | 0 |

Aggregate shell provenance stayed consistent with Milestone 1.0 because learned
generation was not accepted:

| Provenance | Percent |
|---|---:|
| OBSERVED | 0.00% |
| RECONSTRUCTED | 2.02% |
| INFERRED | 12.82% |
| IMAGINED | 76.38% |
| ABSENT | 8.78% |

## Validation

The synthetic learned-inpainting guardrail test passed:

```text
protectedModifiedPixels: 0
criticalViolations: 0
imaginedGeneratablePixels: 9356
failClosedBlocker: unable to load remember/nonexistent-local-model from local cache
```

The private walkthrough run also completed and emitted the Milestone 1.1
metrics/prototype to the session artifact directory, not the repository.

## Verdict

Milestone 1.1 is ready for founder review as a truthful partial: the codebase now
has the generalized learned inpainting surface, hard mask/provenance enforcement,
isolated local dependency setup, demo toggles, metrics, and fail-closed behavior.
It does **not** claim a learned visual result yet because the target model was
not locally cached/available in this environment. The deterministic Milestone
1.0 fallback remains available and is still the default effective engine unless
a validated local model exists.
