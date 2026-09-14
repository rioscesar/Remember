"""Optional local learned inpainting for canonical room-face atlases.

The module is deliberately fail-closed: missing dependencies, missing model
weights, auth/network failures, CUDA/runtime errors, and guardrail violations
return an explicit blocked result instead of fabricating learned output. The
deterministic structural fallback remains the caller's safe default.

Milestone 1.2 adds optional visual memory conditioning via local IP-Adapter.
Contextual memory from ranked source photos may guide generation, but all
generated output remains strictly IMAGINED in provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from evidence_doctrine import ABSENT, IMAGINED, apply_provenance_priority

DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-inpainting"
DEFAULT_IP_ADAPTER_MODEL_ID = "h94/IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter_sd15.safetensors"
_DEFAULT_PACKAGE_PATH_RAW = os.environ.get("REMEMBER_LEARNED_PYTHONPATH", "").strip()
DEFAULT_PACKAGE_PATH = Path(_DEFAULT_PACKAGE_PATH_RAW) if _DEFAULT_PACKAGE_PATH_RAW else None
DEFAULT_NEGATIVE_PROMPT = (
    "people, person, face, body, pet, animal, text, letters, signage, logo, "
    "painting, poster, artwork, framed picture, screen, television, mirror, "
    "personal object, clutter, furniture, decoration, readable content"
)


@dataclass(frozen=True)
class LearnedInpaintingConfig:
    model_id: str = DEFAULT_MODEL_ID
    seed: int = 1101
    steps: int = 24
    guidance_scale: float = 6.0
    max_resolution: int = 512
    allow_model_download: bool = False
    torch_dtype: str = "auto"
    package_path: Path | None = DEFAULT_PACKAGE_PATH
    use_ip_adapter: bool = False
    ip_adapter_model_id: str = DEFAULT_IP_ADAPTER_MODEL_ID
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = DEFAULT_IP_ADAPTER_WEIGHT_NAME
    ip_adapter_scale: float = 0.5
    context_image: np.ndarray | None = None


@dataclass
class LearnedInpaintingResult:
    accepted: bool
    image_bgr: np.ndarray | None
    provenance: np.ndarray | None
    metrics: dict = field(default_factory=dict)
    blocker: str | None = None


def add_optional_package_path(package_path: Path | None = None) -> None:
    """Expose an isolated --target package directory without touching base envs."""
    raw = str(package_path or os.environ.get("REMEMBER_LEARNED_PYTHONPATH", "")).strip()
    if not raw:
        return
    path = Path(raw)
    if path.exists():
        import sys

        sys.path.insert(0, str(path))


def learned_dependencies_status(package_path: Path | None = None) -> dict:
    add_optional_package_path(package_path)
    modules = ("torch", "diffusers", "transformers", "accelerate", "safetensors", "PIL")
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def structural_prompt(face_key: str | None) -> str:
    role = (face_key or "room face").replace("_", " ")
    return (
        f"plain boring empty residential {role}, simple plaster structural surface, "
        "matte paint, subtle even lighting, no objects, no people, no text, no artwork"
    )


def _target_size(width: int, height: int, max_resolution: int) -> tuple[int, int]:
    scale = min(1.0, float(max_resolution) / max(width, height, 1))
    target_w = max(64, int(round(width * scale / 8)) * 8)
    target_h = max(64, int(round(height * scale / 8)) * 8)
    return target_w, target_h


def _memory_used_mb() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except Exception:
        return None
    return None


def _load_pipeline(config: LearnedInpaintingConfig):
    add_optional_package_path(config.package_path)
    missing = [name for name, present in learned_dependencies_status(config.package_path).items() if not present]
    if missing:
        raise RuntimeError(f"missing local learned inpainting dependencies: {', '.join(missing)}")

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    if config.torch_dtype == "float32":
        dtype = torch.float32
    elif config.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=not config.allow_model_download,
        )
    except Exception as exc:
        mode = "local cache" if not config.allow_model_download else "download/auth/cache"
        raise RuntimeError(f"unable to load {config.model_id} from {mode}: {exc}") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.safety_checker = None
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if config.use_ip_adapter:
        try:
            pipe.load_ip_adapter(
                config.ip_adapter_model_id,
                subfolder=config.ip_adapter_subfolder,
                weight_name=config.ip_adapter_weight_name,
                torch_dtype=dtype,
                local_files_only=not config.allow_model_download,
            )
            pipe.set_ip_adapter_scale(config.ip_adapter_scale)
        except Exception as exc:
            mode = "local cache" if not config.allow_model_download else "download/auth/cache"
            raise RuntimeError(
                f"unable to load IP-Adapter {config.ip_adapter_model_id} from {mode}: {exc}"
            ) from exc

    return pipe, torch, device, str(dtype).replace("torch.", "")


def run_learned_inpainting(
    base_color_bgr: np.ndarray,
    base_provenance: np.ndarray,
    generatable_mask: np.ndarray,
    locked_mask: np.ndarray,
    critical_mask: np.ndarray,
    *,
    face_key: str | None = None,
    config: LearnedInpaintingConfig | None = None,
    runner: Callable[[np.ndarray, np.ndarray, str, np.ndarray | None], np.ndarray] | None = None,
) -> LearnedInpaintingResult:
    """Inpaint only GENERATABLE pixels and verify protected pixels are exact.

    `runner` is a test seam that receives resized RGB base, resized inpaint
    mask, prompt, and optional context_rgb, and must return a resized RGB image.
    Production callers leave it unset to use local Diffusers.
    """
    config = config or LearnedInpaintingConfig()
    height, width = base_provenance.shape
    if base_color_bgr.shape[:2] != (height, width):
        raise ValueError("base color and provenance shapes differ")
    for name, mask in (
        ("generatable_mask", generatable_mask),
        ("locked_mask", locked_mask),
        ("critical_mask", critical_mask),
    ):
        if mask.shape != (height, width):
            raise ValueError(f"{name} shape differs from provenance")

    generatable = generatable_mask & ~locked_mask & ~critical_mask
    protected = ~generatable
    if not generatable.any():
        return LearnedInpaintingResult(
            accepted=False,
            image_bgr=None,
            provenance=None,
            metrics={"engine": "learned-diffusion", "accepted": False, "generatablePixels": 0},
            blocker="no generatable atlas pixels",
        )

    started = time.perf_counter()
    target_w, target_h = _target_size(width, height, config.max_resolution)
    prompt = structural_prompt(face_key)
    generated_rgb = None
    model_device = "test-runner"
    precision = "test-runner"
    model_id = config.model_id
    blocker = None

    try:
        base_rgb = cv2.cvtColor(base_color_bgr, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(base_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(
            generatable.astype(np.uint8) * 255,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

        context_rgb = None
        if config.context_image is not None:
            if config.context_image.ndim == 3 and config.context_image.shape[2] == 3:
                # Expecting BGR or RGB; assume BGR if coming from OpenCV
                context_rgb = cv2.cvtColor(config.context_image, cv2.COLOR_BGR2RGB)
            else:
                context_rgb = config.context_image

        if config.use_ip_adapter and context_rgb is None and runner is None:
            raise RuntimeError("IP-Adapter conditioning requested but no context_image was provided")

        if runner is not None:
            # Runner test seam support
            import inspect
            sig = inspect.signature(runner)
            if len(sig.parameters) >= 4:
                generated_rgb = runner(resized_rgb, resized_mask, prompt, context_rgb)
            else:
                generated_rgb = runner(resized_rgb, resized_mask, prompt)
        else:
            try:
                import torch

                model_device = "cuda" if torch.cuda.is_available() else "cpu"
                precision = (
                    config.torch_dtype
                    if config.torch_dtype != "auto"
                    else ("float16" if torch.cuda.is_available() else "float32")
                )
            except Exception:
                model_device = "unavailable"
                precision = config.torch_dtype
            from PIL import Image

            pipe, torch, model_device, precision = _load_pipeline(config)
            generator = torch.Generator(device=model_device).manual_seed(config.seed)

            kwargs = {
                "prompt": prompt,
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                "image": Image.fromarray(resized_rgb),
                "mask_image": Image.fromarray(resized_mask),
                "width": target_w,
                "height": target_h,
                "num_inference_steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "generator": generator,
            }
            if config.use_ip_adapter and context_rgb is not None:
                kwargs["ip_adapter_image"] = Image.fromarray(context_rgb)

            with torch.inference_mode():
                output = pipe(**kwargs)
            generated_rgb = np.asarray(output.images[0].convert("RGB"))
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as exc:
        blocker = str(exc)

    latency_ms = (time.perf_counter() - started) * 1000
    base_metrics = {
        "engine": "learned-diffusion",
        "model": model_id,
        "precision": precision,
        "device": model_device,
        "resolution": {"width": target_w, "height": target_h},
        "steps": config.steps,
        "guidanceScale": config.guidance_scale,
        "seed": config.seed,
        "latencyMs": latency_ms,
        "peakTorchVramMb": _memory_used_mb(),
        "generatablePixels": int(generatable.sum()),
        "lockedPixels": int(locked_mask.sum()),
        "criticalPixels": int(critical_mask.sum()),
        "useIpAdapter": config.use_ip_adapter,
        "ipAdapterModel": config.ip_adapter_model_id if config.use_ip_adapter else None,
        "ipAdapterScale": config.ip_adapter_scale if config.use_ip_adapter else None,
        "hasContextImage": config.context_image is not None,
    }
    if blocker is not None:
        return LearnedInpaintingResult(
            accepted=False,
            image_bgr=None,
            provenance=None,
            metrics={**base_metrics, "accepted": False, "failClosed": True},
            blocker=blocker,
        )

    if generated_rgb is None or generated_rgb.shape[:2] != (target_h, target_w):
        return LearnedInpaintingResult(
            accepted=False,
            image_bgr=None,
            provenance=None,
            metrics={**base_metrics, "accepted": False, "failClosed": True},
            blocker="learned runner returned no image with expected shape",
        )

    generated_bgr = cv2.cvtColor(generated_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    generated_bgr = cv2.resize(generated_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
    candidate_provenance = np.full((height, width), ABSENT, dtype=np.uint8)
    candidate_provenance[generatable] = IMAGINED

    protected_restored = generated_bgr.copy()
    protected_restored[protected] = base_color_bgr[protected]
    output, provenance = apply_provenance_priority(
        base_color_bgr,
        base_provenance,
        protected_restored,
        candidate_provenance,
        locked_mask=locked_mask | critical_mask,
    )

    protected_modified = int(np.any(output[protected] != base_color_bgr[protected], axis=1).sum())
    critical_violations = int(((provenance == IMAGINED) & critical_mask).sum())
    accepted = protected_modified == 0 and critical_violations == 0
    if not accepted:
        return LearnedInpaintingResult(
            accepted=False,
            image_bgr=None,
            provenance=None,
            metrics={
                **base_metrics,
                "accepted": False,
                "failClosed": True,
                "protectedModifiedPixels": protected_modified,
                "criticalViolations": critical_violations,
            },
            blocker="learned output failed mask/provenance guardrails",
        )

    return LearnedInpaintingResult(
        accepted=True,
        image_bgr=output,
        provenance=provenance,
        metrics={
            **base_metrics,
            "accepted": True,
            "failClosed": False,
            "protectedModifiedPixels": protected_modified,
            "criticalViolations": critical_violations,
            "imaginedPixelsAccepted": int((provenance == IMAGINED).sum()),
        },
    )