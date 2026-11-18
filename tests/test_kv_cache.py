"""Context-length KV cache reduces placement budgets."""

from __future__ import annotations

import pytest

from stratumllm import DeploymentStrategyEvaluator, GpuDevice, HardwareProfile, ModelSpec
from stratumllm.exceptions import InsufficientResourcesError
from stratumllm.strategies import FullGPUStrategy, StratifiedOffloadStrategy


def test_kv_cache_estimate_scales_with_context() -> None:
    short = ModelSpec(name="m", parameter_count_b=8.0, num_layers=32, context_length=4096)
    long = short.model_copy(update={"context_length": 32768})
    assert long.estimated_kv_cache_gib() > short.estimated_kv_cache_gib() * 7


def test_long_context_promotes_stratified_on_discrete_gpu() -> None:
    hw = HardwareProfile(
        system_ram_gib=64.0,
        system_ram_available_gib=48.0,
        gpus=[GpuDevice(index=0, name="RTX 4090", vram_gib=24.0)],
        kv_cache_reserve_gib=2.0,
    )
    base = ModelSpec(
        name="llama-13b",
        parameter_count_b=13.0,
        weight_size_gib=14.0,
        num_layers=40,
        context_length=8192,
    )
    strategy_short = DeploymentStrategyEvaluator().evaluate(hw, base)
    assert isinstance(strategy_short, FullGPUStrategy)

    long_ctx = base.model_copy(update={"context_length": 98304})
    strategy_long = DeploymentStrategyEvaluator().evaluate(hw, long_ctx)
    assert isinstance(strategy_long, StratifiedOffloadStrategy)
    assert strategy_long.rationale.kv_cache_gib > strategy_short.rationale.kv_cache_gib
    assert strategy_long.rationale.vram_budget_gib < strategy_short.rationale.vram_budget_gib


def test_long_context_can_fail_unified_pool() -> None:
    hw = HardwareProfile(
        system_ram_gib=16.0,
        system_ram_available_gib=10.0,
        unified_memory=True,
        gpus=[GpuDevice(index=0, name="Apple GPU", vram_gib=16.0)],
        kv_cache_reserve_gib=2.0,
    )
    model = ModelSpec(
        name="llama-8b",
        parameter_count_b=8.0,
        weight_size_gib=8.0,
        num_layers=32,
        context_length=131072,
    )
    with pytest.raises(InsufficientResourcesError):
        DeploymentStrategyEvaluator().evaluate(hw, model)
