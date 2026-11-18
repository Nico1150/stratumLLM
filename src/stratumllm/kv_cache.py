"""KV cache storage assumptions shared by planning and local runtimes."""

from __future__ import annotations

import os
from enum import Enum

from stratumllm.exceptions import ConfigurationError


class KvCacheStorage(str, Enum):
    """
    In-memory K/V tensor dtype for inference (not GGUF weight quant).

    Align with Ollama ``OLLAMA_KV_CACHE_TYPE`` and llama.cpp ``--cache-type-*``.
    """

    F16 = "f16"
    Q8_0 = "q8_0"
    Q4_0 = "q4_0"

    def element_bytes(self) -> float:
        """Bytes per K or V scalar stored for one head position."""
        if self is KvCacheStorage.F16:
            return 2.0
        if self is KvCacheStorage.Q8_0:
            return 1.0
        if self is KvCacheStorage.Q4_0:
            return 0.5
        raise ValueError(f"Unknown storage: {self}")

    def llama_cpp_cache_type(self) -> str:
        """llama.cpp ``--cache-type-k`` / ``--cache-type-v`` token."""
        if self is KvCacheStorage.F16:
            return "f16"
        return self.value


def parse_kv_cache_storage(value: str) -> KvCacheStorage:
    """Parse CLI or ``OLLAMA_KV_CACHE_TYPE`` values."""
    v = value.strip().lower().replace("-", "_")
    if v in ("f16", "fp16"):
        return KvCacheStorage.F16
    if v in ("q8_0", "q8"):
        return KvCacheStorage.Q8_0
    if v in ("q4_0", "q4"):
        return KvCacheStorage.Q4_0
    raise ConfigurationError(
        f"Unknown KV cache storage {value!r}. Use f16, q8_0, or q4_0."
    )


def kv_cache_storage_from_env() -> KvCacheStorage | None:
    """Read ``OLLAMA_KV_CACHE_TYPE`` when set (Ollama global KV quant)."""
    raw = os.environ.get("OLLAMA_KV_CACHE_TYPE", "").strip()
    if not raw:
        return None
    return parse_kv_cache_storage(raw)


def resolve_kv_cache_storage(
    *,
    explicit: KvCacheStorage | None = None,
    runtime: str = "ollama",
) -> KvCacheStorage:
    """
    KV storage for planning aligned with execution.

    Order: explicit (CLI/API) → ``OLLAMA_KV_CACHE_TYPE`` on Ollama → f16.
    """
    if explicit is not None:
        return explicit
    if runtime == "ollama":
        from_env = kv_cache_storage_from_env()
        if from_env is not None:
            return from_env
    return KvCacheStorage.F16


def llama_cpp_cache_cli_args(storage: KvCacheStorage) -> list[str]:
    """Extra ``llama-cli`` flags for K/V cache dtype (quant cache needs flash attn)."""
    if storage is KvCacheStorage.F16:
        return []
    ctype = storage.llama_cpp_cache_type()
    return [
        "--flash-attn",
        "on",
        "--cache-type-k",
        ctype,
        "--cache-type-v",
        ctype,
    ]


def kv_storage_for_runtime(runtime: str) -> KvCacheStorage:
    """Backward-compatible default; prefer ``resolve_kv_cache_storage``."""
    return resolve_kv_cache_storage(runtime=runtime)


def ollama_serve_env_hints(storage: KvCacheStorage) -> dict[str, str]:
    """
    Env vars for ``ollama serve`` so runtime KV matches planning.

    Quantized KV requires Flash Attention in Ollama (see Ollama FAQ).
    """
    hints: dict[str, str] = {
        "OLLAMA_KV_CACHE_TYPE": storage.llama_cpp_cache_type(),
    }
    if storage is not KvCacheStorage.F16:
        hints["OLLAMA_FLASH_ATTENTION"] = "1"
    return hints


def ollama_serve_command(storage: KvCacheStorage) -> str:
    """Single-line example to start Ollama with the planned KV cache dtype."""
    prefix = " ".join(f"{k}={v}" for k, v in ollama_serve_env_hints(storage).items())
    return f"{prefix} ollama serve"


def format_ollama_kv_serve_lines(storage: KvCacheStorage) -> list[str]:
    """Human-readable instructions for CLI (Ollama does not accept KV dtype per request)."""
    lines = [
        "Ollama KV cache is global on the server (not set by StratumLLM /api/generate).",
        f"Planning assumes kv_cache_storage={storage.value}.",
        "Restart the daemon with:",
    ]
    for key, value in ollama_serve_env_hints(storage).items():
        lines.append(f"  export {key}={value}")
    lines.append(f"  # then: ollama serve   (or: {ollama_serve_command(storage)})")
    return lines