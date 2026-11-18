"""Ollama KV plan vs server alignment helpers."""

from __future__ import annotations

from stratumllm.kv_cache import KvCacheStorage, ollama_serve_command, ollama_serve_env_hints
from stratumllm.run.ollama import (
    OllamaKvProbeResult,
    check_ollama_kv_alignment,
    probe_ollama_server_kv_cache_type,
)


def test_ollama_serve_env_hints_quant() -> None:
    env = ollama_serve_env_hints(KvCacheStorage.Q8_0)
    assert env["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert env["OLLAMA_FLASH_ATTENTION"] == "1"


def test_ollama_serve_env_hints_f16() -> None:
    env = ollama_serve_env_hints(KvCacheStorage.F16)
    assert env == {"OLLAMA_KV_CACHE_TYPE": "f16"}
    assert "FLASH" not in ollama_serve_command(KvCacheStorage.F16)


def test_check_aligned_no_warning(monkeypatch) -> None:
    monkeypatch.setattr(
        "stratumllm.run.ollama.probe_ollama_server_kv_cache_type",
        lambda: OllamaKvProbeResult(status="probed", server_storage=KvCacheStorage.Q8_0),
    )
    assert check_ollama_kv_alignment(KvCacheStorage.Q8_0) == []


def test_check_mismatch_warns(monkeypatch) -> None:
    monkeypatch.setattr(
        "stratumllm.run.ollama.probe_ollama_server_kv_cache_type",
        lambda: OllamaKvProbeResult(status="probed", server_storage=KvCacheStorage.F16),
    )
    notes = check_ollama_kv_alignment(KvCacheStorage.Q8_0)
    assert len(notes) == 1
    assert "warning:" in notes[0]
    assert "q8_0" in notes[0]


def test_probe_without_process() -> None:
    result = probe_ollama_server_kv_cache_type()
    assert result.status in ("no_process", "probed", "unreadable")