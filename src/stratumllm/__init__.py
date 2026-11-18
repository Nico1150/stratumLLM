"""StratumLLM: deployment strategy selection for local LLM runtimes."""

from stratumllm.discovery import discover_hardware_profile, hardware_from_dict
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.exceptions import (
    HardwareDiscoveryError,
    InferenceError,
    InsufficientResourcesError,
    ModelResolutionError,
    StratumLLMError,
)
from stratumllm.hardware import GpuDevice, HardwareProfile
from stratumllm.kv_cache import (
    KvCacheStorage,
    parse_kv_cache_storage,
    resolve_kv_cache_storage,
)
from stratumllm.model_spec import ModelSpec, QuantizationKind
from stratumllm.plan_output import (
    build_plan_document,
    build_run_document,
    parse_context_sweep,
    run_context_sweep,
)
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
from stratumllm.run.ollama import (
    check_ollama_kv_alignment,
    ollama_generate,
    probe_ollama_server_kv_cache_type,
)
from stratumllm.run.ollama_env import sync_ollama_kv_from_plan, write_ollama_env_file
from stratumllm.run.ollama_ps import check_ollama_ps_alignment, planned_ollama_resident_gib
from stratumllm.runtime import (
    plan_deployment,
    plan_deployment_from_gguf,
    plan_deployment_from_model,
    run_inference,
)
from stratumllm.strategies import (
    DeploymentStrategyKind,
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)

__all__ = [
    "DeploymentStrategyEvaluator",
    "DeploymentStrategyKind",
    "DistributedStrategy",
    "EvaluatorConfig",
    "FullGPUStrategy",
    "GpuDevice",
    "HardwareDiscoveryError",
    "HardwareProfile",
    "InferenceError",
    "InferenceResult",
    "InsufficientResourcesError",
    "KvCacheStorage",
    "ModelResolutionError",
    "ModelSpec",
    "QuantizationKind",
    "ResolvedModel",
    "StratifiedOffloadStrategy",
    "StratumLLMError",
    "build_plan_document",
    "build_run_document",
    "check_ollama_kv_alignment",
    "check_ollama_ps_alignment",
    "discover_hardware_profile",
    "ensure_model_pulled",
    "hardware_from_dict",
    "llama_cpp_generate",
    "ollama_generate",
    "parse_context_sweep",
    "parse_kv_cache_storage",
    "plan_deployment",
    "plan_deployment_from_gguf",
    "plan_deployment_from_model",
    "planned_ollama_resident_gib",
    "probe_ollama_server_kv_cache_type",
    "resolve_gguf",
    "resolve_gguf_sharded",
    "resolve_kv_cache_storage",
    "resolve_model",
    "resolve_ollama",
    "run_context_sweep",
    "run_inference",
    "strategy_to_ollama_options",
    "sync_ollama_kv_from_plan",
    "write_ollama_env_file",
]
