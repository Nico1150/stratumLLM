"""KV element size tied to runtime storage enum."""

from __future__ import annotations

from stratumllm import ModelSpec
from stratumllm.kv_cache import (
    KvCacheStorage,
    llama_cpp_cache_cli_args,
    parse_kv_cache_storage,
    resolve_kv_cache_storage,
)


def test_f16_element_bytes() -> None:
    assert KvCacheStorage.F16.element_bytes() == 2.0


def test_q4_element_bytes() -> None:
    assert KvCacheStorage.Q4_0.element_bytes() == 0.5


def test_resolve_ollama_defaults_f16(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    assert resolve_kv_cache_storage(runtime="ollama") is KvCacheStorage.F16


def test_resolve_ollama_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    assert resolve_kv_cache_storage(runtime="ollama") is KvCacheStorage.Q8_0


def test_explicit_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
    assert (
        resolve_kv_cache_storage(explicit=KvCacheStorage.Q4_0, runtime="ollama")
        is KvCacheStorage.Q4_0
    )


def test_parse_kv_aliases() -> None:
    assert parse_kv_cache_storage("q8") is KvCacheStorage.Q8_0


def test_llama_cpp_quant_cache_flags() -> None:
    args = llama_cpp_cache_cli_args(KvCacheStorage.Q8_0)
    assert "--flash-attn" in args
    assert "q8_0" in args


def test_llama_cpp_f16_no_extra_flags() -> None:
    assert llama_cpp_cache_cli_args(KvCacheStorage.F16) == []


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