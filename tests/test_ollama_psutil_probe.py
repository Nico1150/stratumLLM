"""psutil-based Ollama KV probe."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stratumllm.kv_cache import KvCacheStorage
from stratumllm.run.ollama import OllamaKvProbeResult, probe_ollama_server_kv_cache_type


def test_probe_uses_psutil_when_available() -> None:
    proc = MagicMock()
    proc.info = {"name": "ollama.exe", "pid": 1}
    proc.name.return_value = "ollama.exe"
    proc.environ.return_value = {"OLLAMA_KV_CACHE_TYPE": "q8_0"}

    with patch("stratumllm.run.ollama.psutil.process_iter", return_value=[proc]):
        result = probe_ollama_server_kv_cache_type()

    assert result == OllamaKvProbeResult(status="probed", server_storage=KvCacheStorage.Q8_0)