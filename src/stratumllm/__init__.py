"""StratumLLM: deployment strategy selection for local LLM runtimes."""

from stratumllm.discovery import discover_hardware_profile, hardware_from_dict
from stratumllm.runtime import plan_deployment
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.exceptions import (
    HardwareDiscoveryError,
    InsufficientResourcesError,
    StratumLLMError,
)
from stratumllm.hardware import GpuDevice, HardwareProfile
from stratumllm.model_spec import ModelSpec, QuantizationKind
from stratumllm.strategies import (
    DeploymentStrategyKind,
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)

__all__ = [
    "DeploymentStrategyEvaluator",
    "EvaluatorConfig",
    "DeploymentStrategyKind",
    "DistributedStrategy",
    "FullGPUStrategy",
    "StratifiedOffloadStrategy",
    "GpuDevice",
    "HardwareDiscoveryError",
    "HardwareProfile",
    "InsufficientResourcesError",
    "ModelSpec",
    "QuantizationKind",
    "StratumLLMError",
    "discover_hardware_profile",
    "hardware_from_dict",
    "plan_deployment",
]
