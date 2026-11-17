"""KV element size tied to runtime storage enum."""

from __future__ import annotations

from stratumllm import ModelSpec
from stratumllm.kv_cache import KvCacheStorage, kv_storage_for_runtime


def test_f16_element_bytes() -> None:
    assert KvCacheStorage.F16.element_bytes() == 2.0


def test_plan_matches_ollama_runtime() -> None:
    assert kv_storage_for_runtime("ollama") is KvCacheStorage.F16


def test_q8_halves_kv_vs_f16() -> None:
    base = ModelSpec(
        name="m",
        parameter_count_b=8.0,
        num_layers=32,
        hidden_size=4096,
        num_kv_heads=8,
        head_dim=128,
        context_length=4096,
        kv_cache_storage=KvCacheStorage.F16,
    )
    q8 = base.model_copy(update={"kv_cache_storage": KvCacheStorage.Q8_0})
    assert q8.estimated_kv_cache_gib() == base.estimated_kv_cache_gib() / 2