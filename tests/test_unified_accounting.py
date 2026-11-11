"""Unified memory planning (single pool, full Metal when feasible)."""

from __future__ import annotations

import pytest

from stratumllm import DeploymentStrategyEvaluator, EvaluatorConfig, HardwareProfile, GpuDevice, ModelSpec
from stratumllm.exceptions import InsufficientResourcesError
from stratumllm.hardware import AcceleratorKind
from stratumllm.strategies import FullGPUStrategy, StratifiedOffloadStrategy


def _mac_like_16g_profile(*, uma_metal_residency_fraction: float = 1.0) -> HardwareProfile:
    return HardwareProfile(
        system_ram_gib=16.0,
        system_ram_available_gib=10.0,
        unified_memory=True,
        uma_metal_residency_fraction=uma_metal_residency_fraction,
        accelerator=AcceleratorKind.METAL,
        gpus=[
            GpuDevice(
                index=0,
                name="Apple GPU",
                vram_gib=16.0,
                vram_free_gib=10.0,
            )
        ],
        gpu_utilization_target=0.90,
        ram_utilization_target=0.75,
        kv_cache_reserve_gib=2.0,
    )


def test_unified_weight_budget_single_pool() -> None:
    hw = _mac_like_16g_profile()
    assert hw.unified_weight_budget_gib() == pytest.approx((16.0 - 2.0) * 0.75)


def test_model_within_pool_gets_full_metal_not_stratified() -> None:
    """8 GiB fits pool (~10.5); must not force partial offload (old 55% slice bug)."""
    hw = _mac_like_16g_profile()
    model = ModelSpec(
        name="llama-8b-q4",
        parameter_count_b=8.0,
        weight_size_gib=8.0,
        num_layers=32,
    )
    strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
    assert isinstance(strategy, FullGPUStrategy)
    assert strategy.host_memory_unified is True
    assert "CUDA_VISIBLE_DEVICES" not in strategy.to_ollama_env()


def test_rejects_weight_over_unified_pool() -> None:
    hw = _mac_like_16g_profile()
    model = ModelSpec(
        name="oversized",
        parameter_count_b=1.0,
        weight_size_gib=17.4,
        num_layers=80,
    )
    with pytest.raises(InsufficientResourcesError):
        DeploymentStrategyEvaluator().evaluate(hw, model)


def test_uma_stratified_only_when_metal_fraction_caps_residency() -> None:
    hw = _mac_like_16g_profile(uma_metal_residency_fraction=0.55)
    model = ModelSpec(
        name="llama-8b-q4",
        parameter_count_b=8.0,
        weight_size_gib=8.0,
        num_layers=80,
    )
    strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
    assert isinstance(strategy, StratifiedOffloadStrategy)
    assert strategy.gpu_layers < strategy.total_layers
    assert strategy.host_memory_unified is True


def test_discrete_scenario_still_stratifies() -> None:
    hw = HardwareProfile(
        system_ram_gib=128.0,
        system_ram_available_gib=96.0,
        unified_memory=False,
        gpus=[GpuDevice(index=0, name="RTX 4090", vram_gib=24.0)],
    )
    m70 = ModelSpec(name="70b", parameter_count_b=70)
    strategy = DeploymentStrategyEvaluator().evaluate(hw, m70)
    assert isinstance(strategy, StratifiedOffloadStrategy)
    assert strategy.host_memory_unified is False
