"""Sync Ollama KV env from --kv-storage."""

from __future__ import annotations

import os

from stratumllm.kv_cache import KvCacheStorage
from stratumllm.run.ollama_env import (
    apply_ollama_kv_to_process,
    apply_ollama_kv_windows,
    sync_ollama_kv_from_plan,
    write_ollama_env_file,
)


def test_apply_ollama_kv_to_process(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_KV_CACHE_TYPE", raising=False)
    monkeypatch.delenv("OLLAMA_FLASH_ATTENTION", raising=False)
    apply_ollama_kv_to_process(KvCacheStorage.Q8_0)
    assert os.environ["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
    assert os.environ["OLLAMA_FLASH_ATTENTION"] == "1"


def test_write_ollama_env_file(tmp_path) -> None:
    path = write_ollama_env_file(tmp_path / "ollama-kv.env", KvCacheStorage.Q8_0)
    text = path.read_text(encoding="utf-8")
    assert "export OLLAMA_KV_CACHE_TYPE=q8_0" in text
    assert "OLLAMA_FLASH_ATTENTION" in text


def test_windows_setx_noop_off_windows(monkeypatch) -> None:
    monkeypatch.setattr("stratumllm.run.ollama_env.platform.system", lambda: "Linux")
    assert apply_ollama_kv_windows(KvCacheStorage.Q8_0) == []


def test_sync_without_restart() -> None:
    result = sync_ollama_kv_from_plan(KvCacheStorage.F16, restart_if_running=False)
    assert result.applied_to_process is True
    assert result.restart_attempted is False
    assert result.windows_setx_ok is True