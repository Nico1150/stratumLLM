"""Resolve on-disk model artifacts into planning-ready specs."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

from stratumllm.exceptions import ModelResolutionError
from stratumllm.model_spec import ModelSpec, QuantizationKind
from stratumllm.resolve.gguf import read_gguf_metadata

# llama.cpp ``general.file_type`` (llama_ftype) → planning quantization.
_FILE_TYPE_TO_QUANT: dict[int, QuantizationKind] = {
    1: QuantizationKind.F16,
    0: QuantizationKind.F16,
    2: QuantizationKind.Q4_0,
    3: QuantizationKind.Q4_0,
    6: QuantizationKind.Q4_K_M,
    7: QuantizationKind.Q4_K_M,
    12: QuantizationKind.Q5_K_M,
    14: QuantizationKind.Q5_K_M,
    15: QuantizationKind.Q4_K_M,
    16: QuantizationKind.Q4_K_M,
    8: QuantizationKind.Q8_0,
    9: QuantizationKind.Q8_0,
}

_LAYER_COUNT_KEYS: tuple[str, ...] = (
    "llama.block_count",
    "gemma.block_count",
    "mistral.block_count",
    "phi3.block_count",
    "qwen2.block_count",
    "qwen.block_count",
    "gptneox.block_count",
    "mpt.block_count",
)

_NAME_PARAM_PATTERN = re.compile(r"(?:^|[\-_/])(\d+(?:\.\d+)?)[bB](?:[\-_.]|$)")


class FootprintSource(str, Enum):
    """Where resolved planning numbers originated."""

    GGUF = "gguf"
    OLLAMA = "ollama"
    MERGED = "merged"
    MANUAL = "manual"


class ResolvedModel(BaseModel):
    """
    Model footprint for deployment planning (truth from artifact + optional overrides).

    Prefer ``weight_size_gib`` from disk; ``parameter_count_b`` is informative only.
    """

    model_config = {"frozen": True}

    name: str = Field(min_length=1)
    weight_size_gib: Annotated[float, Field(gt=0)]
    source: FootprintSource = FootprintSource.GGUF
    artifact_path: Path | None = None
    quantization: QuantizationKind | None = None
    num_layers: Annotated[int | None, Field(default=None, gt=0)] = None
    parameter_count_b: Annotated[float | None, Field(default=None, gt=0)] = None
    architecture: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_model_spec(self, *, context_length: int = 4096) -> ModelSpec:
        """Convert to ``ModelSpec`` for ``DeploymentStrategyEvaluator``."""
        params = self.parameter_count_b if self.parameter_count_b is not None else 1.0
        quant = self.quantization or QuantizationKind.Q4_K_M
        return ModelSpec(
            name=self.name,
            parameter_count_b=params,
            quantization=quant,
            weight_size_gib=self.weight_size_gib,
            num_layers=self.num_layers,
            context_length=context_length,
            architecture=self.architecture or "transformer",
        )


def resolve_gguf(path: str | Path) -> ResolvedModel:
    """
    Build a ``ResolvedModel`` from a local GGUF file.

    Uses on-disk size for weight footprint and GGUF metadata for layers and quantization.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise ModelResolutionError(f"GGUF file not found: {file_path}")
    if file_path.suffix.lower() != ".gguf":
        raise ModelResolutionError(f"Expected a .gguf file, got: {file_path}")
    return _resolve_gguf_file(file_path, source=FootprintSource.GGUF)


def resolve_ollama(
    model_ref: str,
    *,
    expect_parameter_count_b: float | None = None,
    strict: bool = False,
    models_root: Path | None = None,
    pull_if_missing: bool = False,
) -> ResolvedModel:
    """
    Resolve an Ollama model name (e.g. ``qwen2.5:7b``) to a local weights blob.

    Uses Ollama manifests under ``~/.ollama/models`` (or ``OLLAMA_MODELS``).
    Optional ``expect_parameter_count_b`` is checked against GGUF/name/show hints.
    """
    from stratumllm.resolve.ollama import (
        ensure_model_pulled,
        ollama_show_text,
        parse_show_parameters_b,
        resolve_ollama_blob_path,
        verify_parameter_count,
    )

    if pull_if_missing:
        ensure_model_pulled(model_ref)

    blob_path, canonical = resolve_ollama_blob_path(model_ref, models_root=models_root)
    resolved = _resolve_gguf_file(
        blob_path,
        source=FootprintSource.OLLAMA,
        display_name=canonical,
    )
    show_params: float | None = None
    try:
        show_text = ollama_show_text(model_ref)
        show_params = parse_show_parameters_b(show_text)
    except ModelResolutionError:
        show_text = ""

    param_b = resolved.parameter_count_b
    if show_params is not None:
        param_b = show_params
    elif param_b is None or param_b == 1.0:
        param_b = _guess_params_from_name(model_ref) or param_b

    resolved = resolved.model_copy(
        update={
            "name": canonical,
            "parameter_count_b": param_b if param_b is not None else 1.0,
            "metadata": {
                **resolved.metadata,
                "stratum.ollama_ref": model_ref,
                "stratum.ollama_show": show_text[:2000] if show_text else None,
            },
        }
    )

    if expect_parameter_count_b is not None:
        verify_parameter_count(
            resolved.parameter_count_b,
            expect_parameter_count_b,
            strict=strict,
            model_ref=model_ref,
        )
    return resolved


def resolve_model(
    model_ref: str,
    *,
    expect_parameter_count_b: float | None = None,
    strict: bool = False,
    pull_if_missing: bool = False,
) -> ResolvedModel:
    """
    Resolve a model by Ollama name or a local ``.gguf`` path.

    Paths must exist and end with ``.gguf``; otherwise ``model_ref`` is treated
    as an Ollama model name.
    """
    path = Path(model_ref).expanduser()
    if path.suffix.lower() == ".gguf" and path.is_file():
        resolved = resolve_gguf(path)
        if expect_parameter_count_b is not None:
            _verify_expected_params(
                resolved.parameter_count_b,
                expect_parameter_count_b,
                model_ref,
                strict,
            )
        return resolved
    return resolve_ollama(
        model_ref,
        expect_parameter_count_b=expect_parameter_count_b,
        strict=strict,
        pull_if_missing=pull_if_missing,
    )


def _verify_expected_params(
    resolved_b: float | None,
    expected_b: float,
    model_ref: str,
    strict: bool,
) -> None:
    from stratumllm.resolve.ollama import verify_parameter_count

    verify_parameter_count(resolved_b, expected_b, strict=strict, model_ref=model_ref)


def _resolve_gguf_file(
    file_path: Path,
    *,
    source: FootprintSource,
    display_name: str | None = None,
) -> ResolvedModel:
    try:
        metadata = read_gguf_metadata(file_path)
    except (OSError, ValueError) as exc:
        raise ModelResolutionError(f"Failed to parse GGUF metadata: {exc}") from exc

    weight_gib = file_path.stat().st_size / (1024**3)
    name = display_name or _metadata_name(metadata, file_path)
    num_layers = _metadata_layers(metadata)
    quantization = _metadata_quantization(metadata)
    architecture = _metadata_architecture(metadata)
    param_b = _metadata_parameter_count_b(metadata)
    if param_b is None:
        param_b = _guess_params_from_name(name)
    if param_b is None and quantization is not None:
        param_b = _estimate_params_from_weight(weight_gib, quantization)

    return ResolvedModel(
        name=name,
        weight_size_gib=weight_gib,
        source=source,
        artifact_path=file_path,
        quantization=quantization,
        num_layers=num_layers,
        parameter_count_b=param_b if param_b is not None else 1.0,
        architecture=architecture,
        metadata=metadata,
    )


def resolve_gguf_sharded(glob_pattern: str | Path) -> ResolvedModel:
    """
    Resolve one logical model split across multiple GGUF shards (same directory).

    Sums file sizes; metadata is read from the first shard (lowest sort order).
    """
    pattern = Path(glob_pattern).expanduser()
    paths = sorted(pattern.parent.glob(pattern.name))
    if not paths:
        raise ModelResolutionError(f"No GGUF shards match: {glob_pattern}")
    primary = resolve_gguf(paths[0])
    total_gib = sum(p.stat().st_size for p in paths) / (1024**3)
    return primary.model_copy(
        update={
            "weight_size_gib": total_gib,
            "source": FootprintSource.GGUF,
            "metadata": {**primary.metadata, "stratum.shard_count": len(paths), "stratum.shard_paths": [str(p) for p in paths]},
        }
    )


def _metadata_name(metadata: dict[str, Any], path: Path) -> str:
    for key in ("general.name", "general.basename"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return path.stem


def _metadata_layers(metadata: dict[str, Any]) -> int | None:
    for key in _LAYER_COUNT_KEYS:
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for key, value in metadata.items():
        if key.endswith(".block_count") and isinstance(value, int) and value > 0:
            return value
    return None


def _metadata_quantization(metadata: dict[str, Any]) -> QuantizationKind | None:
    file_type = metadata.get("general.file_type")
    if isinstance(file_type, int):
        return _FILE_TYPE_TO_QUANT.get(file_type)
    return None


def _metadata_architecture(metadata: dict[str, Any]) -> str | None:
    value = metadata.get("general.architecture")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _metadata_parameter_count_b(metadata: dict[str, Any]) -> float | None:
    for key in (
        "general.parameter_count",
        "llama.parameter_count",
        "gemma.parameter_count",
    ):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return value / 1e9
        if isinstance(value, float) and value > 0:
            return value / 1e9 if value > 1e6 else value
    return None


def _guess_params_from_name(name: str) -> float | None:
    match = _NAME_PARAM_PATTERN.search(name)
    if not match:
        return None
    return float(match.group(1))


def _estimate_params_from_weight(weight_gib: float, quantization: QuantizationKind) -> float:
    from stratumllm.model_spec import bytes_per_parameter

    bytes_total = weight_gib * (1024**3) / 1.05
    bpp = bytes_per_parameter(quantization)
    params = bytes_total / bpp
    return max(0.1, params / 1e9)
