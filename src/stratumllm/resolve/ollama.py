"""Resolve Ollama model names to local GGUF blobs via manifests."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from stratumllm.exceptions import ModelResolutionError
from stratumllm.model_spec import QuantizationKind

OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"
_PARAM_TOLERANCE = 0.12

_SHOW_PARAMETERS = re.compile(r"^\s*parameters\s+(\d+(?:\.\d+)?)\s*([bB])\s*$", re.MULTILINE)
_SHOW_CONTEXT_LENGTH = re.compile(
    r"^\s*context\s+length\s+(\d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_SHOW_QUANT = re.compile(r"^\s*quantization\s+(\S+)\s*$", re.MULTILINE)
_SHOW_ARCH = re.compile(r"^\s*architecture\s+(\S+)\s*$", re.MULTILINE)


def ollama_models_root() -> Path:
    env = __import__("os").environ.get("OLLAMA_MODELS")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".ollama" / "models"


def parse_ollama_model_ref(model_ref: str) -> tuple[str, str]:
    """
    Split ``qwen2.5:7b`` → (``qwen2.5``, ``7b``); default tag ``latest``.
    """
    ref = model_ref.strip()
    if not ref:
        raise ModelResolutionError("Empty Ollama model name.")
    if ":" in ref:
        base, tag = ref.split(":", 1)
        return base, tag or "latest"
    return ref, "latest"


def manifest_path_for_ref(models_root: Path, model_ref: str) -> Path:
    base, tag = parse_ollama_model_ref(model_ref)
    return models_root / "manifests" / "registry.ollama.ai" / "library" / base / tag


def find_model_blob_digest(manifest_path: Path) -> tuple[str, int]:
    """Return (blob filename suffix hex, layer size bytes) for the weights layer."""
    if not manifest_path.is_file():
        raise ModelResolutionError(f"Ollama manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelResolutionError(f"Invalid Ollama manifest: {manifest_path}") from exc

    layers: list[dict[str, Any]] = data.get("layers", [])
    for layer in layers:
        if layer.get("mediaType") == OLLAMA_MODEL_LAYER:
            digest = str(layer.get("digest", ""))
            if not digest.startswith("sha256:"):
                raise ModelResolutionError("Unexpected Ollama layer digest.")
            hex_id = digest.removeprefix("sha256:")
            size = int(layer.get("size", 0))
            return hex_id, size
    raise ModelResolutionError(f"No model weights layer in {manifest_path}")


def blob_path(models_root: Path, digest_hex: str) -> Path:
    return models_root / "blobs" / f"sha256-{digest_hex}"


def candidate_manifests(models_root: Path, model_ref: str) -> list[Path]:
    base, tag = parse_ollama_model_ref(model_ref)
    paths = [
        models_root / "manifests" / "registry.ollama.ai" / "library" / base / tag,
    ]
    if tag != "latest":
        paths.append(models_root / "manifests" / "registry.ollama.ai" / "library" / base / "latest")
    return paths


def ensure_model_pulled(model_ref: str, *, pull_if_missing: bool = True) -> None:
    """
    Ensure ``model_ref`` exists locally for Ollama (``ollama show`` / ``ollama pull``).

    Call before resolving manifests when the model may not be downloaded yet.
    """
    if shutil.which("ollama") is None:
        raise ModelResolutionError("`ollama` CLI not found on PATH.")
    try:
        result = subprocess.run(
            ["ollama", "show", model_ref],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ModelResolutionError(f"Failed to query Ollama model {model_ref!r}.") from exc
    if result.returncode == 0:
        return
    if not pull_if_missing:
        raise ModelResolutionError(
            f"Ollama model {model_ref!r} is not local. Run `ollama pull {model_ref}`."
        )
    try:
        subprocess.run(
            ["ollama", "pull", model_ref],
            check=True,
            timeout=3600,
        )
    except subprocess.CalledProcessError as exc:
        raise ModelResolutionError(f"`ollama pull {model_ref}` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ModelResolutionError(f"`ollama pull {model_ref}` timed out.") from exc


def resolve_ollama_blob_path(
    model_ref: str,
    *,
    models_root: Path | None = None,
) -> tuple[Path, str]:
    """
    Locate the on-disk GGUF blob for an Ollama model reference.

    Returns:
        Tuple of blob path and canonical ref used for planning labels.
    """
    root = models_root or ollama_models_root()
    if not root.is_dir():
        raise ModelResolutionError(
            f"Ollama models directory not found at {root}. "
            "Install Ollama and run `ollama pull <model>` first."
        )

    last_error: Exception | None = None
    for manifest in candidate_manifests(root, model_ref):
        try:
            digest_hex, _size = find_model_blob_digest(manifest)
            path = blob_path(root, digest_hex)
            if not path.is_file():
                raise ModelResolutionError(f"Ollama blob missing: {path}")
            base, tag = parse_ollama_model_ref(model_ref)
            canonical = f"{base}:{tag}" if tag != "latest" else base
            return path, canonical
        except ModelResolutionError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise ModelResolutionError(
            f"Could not resolve Ollama model {model_ref!r}. "
            f"Try `ollama pull {model_ref}`."
        ) from last_error
    raise ModelResolutionError(f"Could not resolve Ollama model {model_ref!r}.")


def ollama_show_text(model_ref: str, *, timeout_s: float = 30.0) -> str:
    """Run ``ollama show`` for supplemental metadata (best-effort)."""
    try:
        result = subprocess.run(
            ["ollama", "show", model_ref],
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise ModelResolutionError("`ollama` CLI not found on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise ModelResolutionError(
            f"`ollama show` failed for {model_ref!r}: {exc.stderr.strip() or exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ModelResolutionError(f"`ollama show` timed out for {model_ref!r}.") from exc
    return result.stdout


def parse_show_parameters_b(show_text: str) -> float | None:
    match = _SHOW_PARAMETERS.search(show_text)
    if not match:
        return None
    value = float(match.group(1))
    return value


def parse_show_context_length(show_text: str) -> int | None:
    """Parse ``context length`` line from ``ollama show`` output."""
    match = _SHOW_CONTEXT_LENGTH.search(show_text)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def parse_show_quantization_label(show_text: str) -> str | None:
    """Parse ``quantization`` line from ``ollama show`` output."""
    match = _SHOW_QUANT.search(show_text)
    if not match:
        return None
    label = match.group(1).strip()
    return label if label else None


_SHOW_QUANT_TO_KIND: dict[str, QuantizationKind] = {
    "q4_k_m": QuantizationKind.Q4_K_M,
    "q4_0": QuantizationKind.Q4_0,
    "q5_k_m": QuantizationKind.Q5_K_M,
    "q8_0": QuantizationKind.Q8_0,
    "f16": QuantizationKind.F16,
    "fp16": QuantizationKind.F16,
}


def quantization_from_show(show_text: str) -> QuantizationKind | None:
    """Map ``ollama show`` quantization label to ``QuantizationKind``."""
    label = parse_show_quantization_label(show_text)
    if label is None:
        return None
    key = label.lower().replace("-", "_")
    return _SHOW_QUANT_TO_KIND.get(key)


def verify_parameter_count(
    resolved_b: float | None,
    expected_b: float,
    *,
    strict: bool,
    model_ref: str,
) -> None:
    if resolved_b is None:
        if strict:
            raise ModelResolutionError(
                f"Cannot verify parameter count for {model_ref!r}: no resolved count."
            )
        return
    if expected_b <= 0:
        raise ModelResolutionError("expected_parameter_count_b must be positive.")
    delta = abs(resolved_b - expected_b) / expected_b
    if delta > _PARAM_TOLERANCE:
        msg = (
            f"Parameter mismatch for {model_ref!r}: "
            f"expected {expected_b}B, resolved {resolved_b:.2f}B."
        )
        if strict:
            raise ModelResolutionError(msg)
        # Non-strict: caller may log; footprint still from GGUF.
