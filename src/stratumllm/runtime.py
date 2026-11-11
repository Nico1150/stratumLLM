"""High-level API: discover hardware, evaluate model, export runtime kwargs."""

from __future__ import annotations

from stratumllm.discovery import discover_hardware_profile
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.hardware import HardwareProfile
from stratumllm.model_spec import ModelSpec
from stratumllm.strategies import DeploymentStrategy


def plan_deployment(
    model: ModelSpec,
    hardware: HardwareProfile | None = None,
    *,
    evaluator_config: EvaluatorConfig | None = None,
) -> DeploymentStrategy:
    """
    End-to-end planning entry point for integrators.

    Args:
        model: Target LLM specification.
        hardware: Optional pre-validated profile; auto-discovered if omitted.
        evaluator_config: Optional tuning for distributed / CPU-only behavior.

    Returns:
        A strategy profile with ``to_ollama_*`` and ``to_llama_cpp_kwargs`` exporters.
    """
    hw = hardware or discover_hardware_profile()
    evaluator = DeploymentStrategyEvaluator(evaluator_config)
    return evaluator.evaluate(hw, model)
