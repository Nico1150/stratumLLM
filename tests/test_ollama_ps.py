"""Ollama /api/ps alignment helpers."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from stratumllm.run.ollama_ps import (
    check_ollama_ps_alignment,
    planned_ollama_resident_gib,
)
from stratumllm.strategies import FullGPUStrategy, StrategyRationale


def _strategy() -> FullGPUStrategy:
    return FullGPUStrategy(
        device_index=0,
        device_name="gpu",
        rationale=StrategyRationale(
            summary="t",
            weight_gib=4.0,
            vram_budget_gib=8.0,
            ram_budget_gib=16.0,
            kv_cache_gib=1.0,
        ),
    )


def test_planned_resident_gib() -> None:
    assert planned_ollama_resident_gib(_strategy()) == 5.0


def test_ps_context_mismatch_warns() -> None:
    body = json.dumps(
        {
            "models": [
                {
                    "name": "demo",
                    "size": 6 * 1024**3,
                    "size_vram": 6 * 1024**3,
                    "context_length": 2048,
                }
            ]
        }
    ).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        notes = check_ollama_ps_alignment(
            "demo",
            _strategy(),
            planned_context=4096,
            strict=False,
        )

    assert any("context_length" in line for line in notes)