"""High-level API: discover hardware, evaluate model, export runtime kwargs."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

from stratumllm.discovery import discover_hardware_profile
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.hardware import HardwareProfile
from stratumllm.kv_cache import KvCacheStorage, resolve_kv_cache_storage
from stratumllm.model_spec import ModelSpec
from stratumllm.resolve import ResolvedModel, resolve_gguf, resolve_model
from stratumllm.run import InferenceResult
from stratumllm.run.llama_cpp import llama_cpp_generate
from stratumllm.run.ollama import (
    check_ollama_kv_alignment,
    ensure_ollama_available,
    ollama_generate,
)
from stratumllm.run.ollama_ps import check_ollama_ps_alignment
from stratumllm.strategies import DeploymentStrategy

RuntimeKind = Literal["ollama", "llama.cpp", "auto"]


def infer_runtime_kind(model_ref: str, runtime: RuntimeKind = "auto") -> RuntimeKind:
    """Resolve ``auto``: ``.gguf`` paths → llama.cpp, else Ollama."""
    if runtime != "auto":
        return runtime
    path = Path(model_ref).expanduser()
    if path.suffix.lower() == ".gguf" and path.is_file():
        return "llama.cpp"
    return "ollama"


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


def plan_deployment_from_model(
    model_ref: str,
    hardware: HardwareProfile | None = None,
    *,
    evaluator_config: EvaluatorConfig | None = None,
    context_length: int | None = None,
    expect_parameter_count_b: float | None = None,
    strict: bool = False,
    pull_if_missing: bool = False,
    runtime: RuntimeKind = "auto",
    kv_cache_storage: KvCacheStorage | None = None,
) -> tuple[ResolvedModel, DeploymentStrategy]:
    """Resolve Ollama name or GGUF path, then plan deployment."""
    resolved = resolve_model(
        model_ref,
        expect_parameter_count_b=expect_parameter_count_b,
        strict=strict,
        pull_if_missing=pull_if_missing,
    )
    kind = infer_runtime_kind(model_ref, runtime)
    storage = resolve_kv_cache_storage(explicit=kv_cache_storage, runtime=kind)
    model = resolved.to_model_spec(
        context_length=context_length,
        kv_cache_storage=storage,
    )
    return resolved, plan_deployment(model, hardware, evaluator_config=evaluator_config)


def plan_deployment_from_gguf(
    gguf_path: str | Path,
    hardware: HardwareProfile | None = None,
    *,
    evaluator_config: EvaluatorConfig | None = None,
    context_length: int | None = None,
    runtime: RuntimeKind = "llama.cpp",
    kv_cache_storage: KvCacheStorage | None = None,
) -> tuple[ResolvedModel, DeploymentStrategy]:
    """
    Resolve a local GGUF artifact, then plan deployment.

    Returns:
        Tuple of resolved footprint (audit trail) and strategy.
    """
    resolved = resolve_gguf(gguf_path)
    storage = resolve_kv_cache_storage(explicit=kv_cache_storage, runtime=runtime)
    model = resolved.to_model_spec(
        context_length=context_length,
        kv_cache_storage=storage,
    )
    return resolved, plan_deployment(model, hardware, evaluator_config=evaluator_config)


def run_inference(
    model_ref: str,
    prompt: str,
    *,
    hardware: HardwareProfile | None = None,
    evaluator_config: EvaluatorConfig | None = None,
    runtime: RuntimeKind = "auto",
    pull_if_missing: bool = True,
    expect_parameter_count_b: float | None = None,
    strict: bool = False,
    context_length: int | None = None,
    kv_cache_storage: KvCacheStorage | None = None,
    restart_ollama: bool = False,
    check_ollama_kv: bool | None = None,
    strict_ollama_kv: bool = False,
    check_ollama_ps: bool = False,
    strict_ollama_ps: bool = False,
    max_tokens: int = 256,
    ollama_base_url: str | None = None,
) -> tuple[ResolvedModel, DeploymentStrategy, InferenceResult]:
    """
    Plan deployment, then run inference with Ollama or llama-cli.

    ``runtime="auto"``: Ollama for non-.gguf refs, llama-cli for ``.gguf`` paths.

    For Ollama names, downloads the model with ``ollama pull`` when missing *before*
    resolve/plan, so blob-based planning can run.
    """
    kind = infer_runtime_kind(model_ref, runtime)
    use_ollama = kind == "ollama"

    resolved, strategy = plan_deployment_from_model(
        model_ref,
        hardware,
        evaluator_config=evaluator_config,
        context_length=context_length,
        expect_parameter_count_b=expect_parameter_count_b,
        strict=strict,
        pull_if_missing=pull_if_missing and use_ollama,
        runtime=kind,
        kv_cache_storage=kv_cache_storage,
    )
    storage = resolve_kv_cache_storage(explicit=kv_cache_storage, runtime=kind)
    num_ctx = (
        context_length
        if context_length is not None
        else (resolved.context_length or 4096)
    )

    if not use_ollama:
        result = llama_cpp_generate(
            resolved,
            prompt,
            strategy,
            max_tokens=max_tokens,
            num_ctx=num_ctx,
            kv_cache_storage=storage,
        )
        return resolved, strategy, result

    if kv_cache_storage is not None:
        from stratumllm.run.ollama_env import sync_ollama_kv_from_plan

        sync_ollama_kv_from_plan(
            storage,
            restart_if_running=restart_ollama,
            base_url=ollama_base_url,
        )

    ensure_ollama_available(ollama_base_url)
    do_check_kv = (
        check_ollama_kv
        if check_ollama_kv is not None
        else kv_cache_storage is not None
    )
    if do_check_kv or strict_ollama_kv:
        for line in check_ollama_kv_alignment(storage, strict=strict_ollama_kv):
            warnings.warn(line, UserWarning, stacklevel=2)
    result = ollama_generate(
        model_ref,
        prompt,
        strategy,
        base_url=ollama_base_url,
        num_ctx=num_ctx,
    )
    if check_ollama_ps or strict_ollama_ps:
        for line in check_ollama_ps_alignment(
            model_ref,
            strategy,
            planned_context=num_ctx,
            base_url=ollama_base_url,
            strict=strict_ollama_ps,
        ):
            warnings.warn(line, UserWarning, stacklevel=2)
    return resolved, strategy, result
