"""Plan output helpers."""

from __future__ import annotations

import pytest

from stratumllm import ModelSpec, discover_hardware_profile
from stratumllm.exceptions import ConfigurationError
from stratumllm.kv_cache import KvCacheStorage
from stratumllm.plan_output import parse_context_sweep, run_context_sweep


def test_parse_context_sweep_list() -> None:
    assert parse_context_sweep("4096, 8192, 16384") == [4096, 8192, 16384]


def test_parse_context_sweep_range() -> None:
    assert parse_context_sweep("4096:12288:4096") == [4096, 8192, 12288]


def test_parse_context_sweep_invalid() -> None:
    with pytest.raises(ConfigurationError):
        parse_context_sweep("4096:8192")


def test_run_context_sweep_manual() -> None:
    hw = discover_hardware_profile()
    base = ModelSpec(
        name="tiny",
        parameter_count_b=0.5,
        context_length=4096,
        weight_size_gib=0.3,
        num_layers=8,
        hidden_size=512,
        num_kv_heads=4,
        head_dim=64,
        kv_cache_storage=KvCacheStorage.F16,
    )
    rows = run_context_sweep(
        hw,
        resolved=None,
        manual_spec=base,
        storage=KvCacheStorage.F16,
        contexts=[4096, 32768],
    )
    assert len(rows) == 2
    assert rows[1]["kv_cache_gib"] > rows[0]["kv_cache_gib"]
    assert rows[0]["strategy"] or rows[0].get("error")