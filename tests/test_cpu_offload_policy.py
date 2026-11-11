"""CPU offload policy caps (GiB and fraction of model)."""

from __future__ import annotations

import pytest

from stratumllm import DeploymentStrategyEvaluator, EvaluatorConfig, HardwareProfile, GpuDevice, ModelSpec
from stratumllm.exceptions import InsufficientResourcesError
from stratumllm.strategies import StratifiedOffloadStrategy


def _discrete_24g_128ram() -> HardwareProfile:
    return HardwareProfile(
        system_ram_gib=128.0,
        system_ram_available_gib=96.0,
        unified_memory=False,
        gpus=[GpuDevice(index=0, name="RTX 4090", vram_gib=24.0)],
    )


def test_max_cpu_offload_fraction_respected_when_feasible() -> None:
    hw = _discrete_24g_128ram()
    m70 = ModelSpec(name="70b", parameter_count_b=70)
    weight = m70.estimated_weight_gib()
    fraction = 0.55

    strategy = DeploymentStrategyEvaluator(
        EvaluatorConfig(max_cpu_offload_fraction=fraction)
    ).evaluate(hw, m70)
    assert isinstance(strategy, StratifiedOffloadStrategy)
    assert strategy.cpu_weight_gib <= weight * fraction + 0.6


def test_max_cpu_offload_gib_rejects_when_gpu_cannot_compensate() -> None:
    hw = _discrete_24g_128ram()
    m70 = ModelSpec(name="70b", parameter_count_b=70)
    with pytest.raises(InsufficientResourcesError):
        DeploymentStrategyEvaluator(EvaluatorConfig(max_cpu_offload_gib=1.0)).evaluate(hw, m70)


def test_max_cpu_only_weight_gib() -> None:
    hw = HardwareProfile(system_ram_gib=64.0, system_ram_available_gib=48.0, gpus=[])
    tiny = ModelSpec(name="tiny", parameter_count_b=1.0, weight_size_gib=3.0, num_layers=8)
    big = ModelSpec(name="big", parameter_count_b=8.0, weight_size_gib=30.0, num_layers=32)
    ev = DeploymentStrategyEvaluator(
        EvaluatorConfig(allow_cpu_only=True, max_cpu_only_weight_gib=8.0)
    )
    ev.evaluate(hw, tiny)
    with pytest.raises(InsufficientResourcesError):
        ev.evaluate(hw, big)
