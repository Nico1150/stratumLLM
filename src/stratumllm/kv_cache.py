"""KV cache storage assumptions shared by planning and local runtimes."""

from __future__ import annotations

from enum import Enum


class KvCacheStorage(str, Enum):
    """
    In-memory K/V tensor dtype for inference (not GGUF weight quant).

    StratumLLM does not read this from GGUF; it must match what the runner uses.
    """

    F16 = "f16"
    Q8_0 = "q8_0"

    def element_bytes(self) -> float:
        """Bytes per K or V scalar stored for one head position."""
        if self is KvCacheStorage.F16:
            return 2.0
        if self is KvCacheStorage.Q8_0:
            return 1.0
        raise ValueError(f"Unknown storage: {self}")


def kv_storage_for_runtime(runtime: str) -> KvCacheStorage:
    """
    KV storage for planning aligned with execution.

    Ollama and llama.cpp both use llama.cpp; default K/V cache is f16 unless you
    opt in to quantized cache via runner flags (not wired yet).
    """
    if runtime in ("ollama", "llama.cpp", "auto", "plan"):
        return KvCacheStorage.F16
    return KvCacheStorage.F16