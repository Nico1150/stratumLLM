"""Attention / KV sizing fields from GGUF metadata (llama.cpp key layout)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _positive_int(value: Any) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    return None


class AttentionFootprint(BaseModel):
    """Optional architecture numbers for KV cache planning."""

    model_config = {"frozen": True}

    hidden_size: int | None = Field(default=None, gt=0)
    num_kv_heads: int | None = Field(default=None, gt=0)
    head_dim: int | None = Field(default=None, gt=0)


def attention_footprint_from_gguf(metadata: dict[str, Any]) -> AttentionFootprint:
    """
    Read embedding width, GQA KV heads, and head dim from GGUF KV metadata.

    Uses ``general.architecture`` as the primary prefix (e.g. ``llama``, ``qwen2``),
    then suffix scans for ``*.embedding_length``, ``*.attention.head_count*``, etc.
    """
    arch = metadata.get("general.architecture")
    prefixes: list[str] = []
    if isinstance(arch, str) and arch.strip():
        prefixes.append(arch.strip())
    for fallback in ("llama", "qwen2", "qwen", "gemma", "mistral", "phi3", "gptneox", "mpt"):
        if fallback not in prefixes:
            prefixes.append(fallback)

    hidden: int | None = None
    head_count: int | None = None
    head_count_kv: int | None = None
    key_length: int | None = None

    for prefix in prefixes:
        if hidden is None:
            hidden = _positive_int(metadata.get(f"{prefix}.embedding_length"))
        if head_count is None:
            head_count = _positive_int(metadata.get(f"{prefix}.attention.head_count"))
        if head_count_kv is None:
            head_count_kv = _positive_int(metadata.get(f"{prefix}.attention.head_count_kv"))
        if key_length is None:
            key_length = _positive_int(metadata.get(f"{prefix}.attention.key_length"))
            if key_length is None:
                key_length = _positive_int(metadata.get(f"{prefix}.attention.value_length"))

    if hidden is None:
        for key, value in metadata.items():
            if key.endswith(".embedding_length"):
                hidden = _positive_int(value)
                if hidden is not None:
                    break

    if head_count is None:
        for key, value in metadata.items():
            if key.endswith(".attention.head_count") and not key.endswith("_kv"):
                head_count = _positive_int(value)
                if head_count is not None:
                    break

    if head_count_kv is None:
        for key, value in metadata.items():
            if key.endswith(".attention.head_count_kv"):
                head_count_kv = _positive_int(value)
                if head_count_kv is not None:
                    break

    if key_length is None:
        for key, value in metadata.items():
            if key.endswith((".attention.key_length", ".attention.value_length")):
                key_length = _positive_int(value)
                if key_length is not None:
                    break

    head_dim = key_length
    if (
        head_dim is None
        and hidden is not None
        and head_count is not None
        and hidden % head_count == 0
    ):
        head_dim = hidden // head_count

    # GQA: ``head_count_kv`` < ``head_count``. MHA: key omitted → all query heads store KV.
    num_kv_heads = head_count_kv if head_count_kv is not None else head_count

    return AttentionFootprint(
        hidden_size=hidden,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )


def context_length_from_gguf(metadata: dict[str, Any]) -> int | None:
    """Read trained context window from GGUF (``{arch}.context_length`` or suffix scan)."""
    arch = metadata.get("general.architecture")
    prefixes: list[str] = []
    if isinstance(arch, str) and arch.strip():
        prefixes.append(arch.strip())
    for fallback in ("llama", "qwen2", "qwen", "gemma", "mistral", "phi3", "gptneox"):
        if fallback not in prefixes:
            prefixes.append(fallback)

    for prefix in prefixes:
        ctx = _positive_int(metadata.get(f"{prefix}.context_length"))
        if ctx is not None:
            return ctx

    for key in ("general.context_length", "general.training.context_length"):
        ctx = _positive_int(metadata.get(key))
        if ctx is not None:
            return ctx

    for key, value in metadata.items():
        if key.endswith(".context_length"):
            ctx = _positive_int(value)
            if ctx is not None:
                return ctx
    return None