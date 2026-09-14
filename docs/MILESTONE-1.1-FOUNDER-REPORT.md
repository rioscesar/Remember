# Founder steering update — Milestone 1.1A

Milestone 1.1A validated one local learned inpainting comparison in canonical
room-face atlas space. It did not change reconstruction, Android, AR, prompts,
objects, the model family, or the deterministic fallback. The model, private
reconstruction inputs, and rendered artifacts remain outside Git.

## Runtime and model

| Item | Result |
|---|---|
| Python executable | `C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\radiance-env\python310\python.exe` |
| Python / pip | 3.10.11 / pip 26.2.1 |
| Torch / CUDA | 2.1.2+cu118 / CUDA 11.8 |
| GPU / total VRAM | NVIDIA GeForce RTX 3070 Laptop GPU / 8,191.5 MiB |
| Diffusers / Transformers | 0.30.3 / 4.44.2 |
| Accelerate / Safetensors | 0.33.0 / 0.4.5 |
| Model | `stable-diffusion-v1-5/stable-diffusion-inpainting` |
| Model cache | `C:\Users\riosc\.cache\huggingface` (27 files, 15,230,241,054 bytes) |

The shell `python` command resolves only to the Windows Store alias. The
companion uses the isolated Python executable above and its existing CUDA
Torch. Diffusers, Transformers, Accelerate, and Safetensors were already
available from the isolated learned-package directory; no runtime dependency
or Torch package was changed.

## Gate results

The fixed-seed synthetic fp16 GPU smoke test used the production
`run_learned_inpainting` path at 64 x 64, seed `1101`, one step, and guidance
`6.0`. It executed on CUDA in float16, took **5,424.42 ms**, used **2,787 MiB**
peak Torch VRAM, accepted **2,880** learned pixels, and recorded
`protectedModifiedPixels: 0` and `criticalViolations: 0`. CPU-only execution
was explicitly rejected.

The existing learned-inpainting guardrail test passed with:

```text
protectedModifiedPixels: 0
criticalViolations: 0
imaginedGeneratablePixels: 9356
```

The production pipeline now supplies its calculated atlas width and height to
Diffusers. Without those arguments, Diffusers defaulted small non-square
inputs to 512 x 512, which made the output-shape guardrail reject otherwise
valid output.

## Single private learned-face comparison

Exactly one real missing canonical face was processed: `back`. The run used
the existing dense evidence PLY and visibility sidecar, 24 steps, guidance
`6.0`, seed `1101`, the required model, and fp16 CUDA. No other missing face
was generated (`generatedFacesCount: 1`); object-gap generation remained `0`.

| Measurement | Result |
|---|---:|
| Model input resolution | 480 x 160 |
| Canonical atlas output | 480 x 158 |
| Learned latency | 8,304.27 ms |
| Peak Torch VRAM | 2,826 MiB |
| Learned accepted pixels | 75,840 |
| Protected modifications | 0 |
| Critical violations | 0 |
| Pipeline generation latency | 8,940.99 ms |
| `nvidia-smi` VRAM before / after | 243 MiB / 435 MiB |

Private artifacts are at
`C:\Users\riosc\.copilot\session-state\f7b0095f-4374-4b1c-a655-bdbc61baba17\files\milestone11a-learned-comparison`:
Remember, deterministic Imagine, learned Imagine, input, mask, candidate,
protected composite, and provenance for `back`, plus the private prototype and
metrics. They were not added to Git.

## Verdict

**B — validated partial.** The learned GPU path, hard mask/provenance
guardrails, and one real canonical-atlas face comparison passed. This is not
an A verdict because the milestone deliberately stopped after that single
comparison and does not establish multi-face learned completion quality.
