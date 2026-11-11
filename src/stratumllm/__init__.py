"""StratumLLM: deployment strategy selection for local LLM runtimes."""

from stratumllm.discovery import discover_hardware_profile, hardware_from_dict
from stratumllm.resolve import (
    ResolvedModel,
    resolve_gguf,
    resolve_gguf_sharded,
    resolve_model,
    resolve_ollama,
)
from stratumllm.resolve.ollama import ensure_model_pulled
from stratumllm.run import InferenceResult, strategy_to_ollama_options
from stratumllm.run.llama_cpp import llama_cpp_generate
from stratumllm.run.ollama import ollama_generate
from stratumllm.runtime import plan_deployment, plan_deployment_from_gguf, plan_deployment_from_model, run_inference
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.exceptions import (
    HardwareDiscoveryError,
    InferenceError,
    InsufficientResourcesError,
    ModelResolutionError,
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
    "InferenceError",
    "InferenceResult",
    "InsufficientResourcesError",
    "ModelResolutionError",
    "ModelSpec",
    "QuantizationKind",
    "ResolvedModel",
    "StratumLLMError",
    "discover_hardware_profile",
    "ensure_model_pulled",
    "hardware_from_dict",
    "plan_deployment",
    "plan_deployment_from_gguf",
    "plan_deployment_from_model",
    "resolve_gguf",
    "resolve_gguf_sharded",
    "resolve_model",
    "resolve_ollama",
    "run_inference",
    "ollama_generate",
    "llama_cpp_generate",
    "strategy_to_ollama_options",
]
