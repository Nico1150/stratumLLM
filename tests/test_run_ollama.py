"""Ollama runner unit tests (HTTP mocked)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from stratumllm.strategies import FullGPUStrategy, StrategyRationale
from stratumllm.run.ollama import ollama_generate


def _full_gpu_strategy() -> FullGPUStrategy:
    return FullGPUStrategy(
        device_index=0,
        device_name="gpu",
        rationale=StrategyRationale(
            summary="test",
            weight_gib=4.0,
            vram_budget_gib=4.0,
            ram_budget_gib=8.0,
        ),
    )


def test_ollama_generate_parses_response() -> None:
    payload = json.dumps({"response": "hello", "done": True}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = payload
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp) as urlopen:
        result = ollama_generate("demo", "hi", _full_gpu_strategy())

    assert result.text == "hello"
    assert result.runtime == "ollama"
    assert urlopen.called
