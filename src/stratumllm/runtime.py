"""High-level API: discover hardware, evaluate model, export runtime kwargs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from stratumllm.discovery import discover_hardware_profile
from stratumllm.evaluator import DeploymentStrategyEvaluator, EvaluatorConfig
from stratumllm.hardware import HardwareProfile
from stratumllm.model_spec import ModelSpec
from stratumllm.resolve import ResolvedModel, resolve_gguf, resolve_model
from stratumllm.run import InferenceResult
from stratumllm.run.llama_cpp import llama_cpp_generate
from stratumllm.run.ollama import (
    ensure_model_pulled,
    ensure_ollama_available,
    ollama_generate,
)
from stratumllm.strategies import DeploymentStrategy

RuntimeKind = Literal["ollama", "llama.cpp", "auto"]


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
    context_length: int = 4096,
    expect_parameter_count_b: float | None = None,
    strict: bool = False,
    pull_if_missing: bool = False,
) -> tuple[ResolvedModel, DeploymentStrategy]:
    """Resolve Ollama name or GGUF path, then plan deployment."""
    resolved = resolve_model(
        model_ref,
        expect_parameter_count_b=expect_parameter_count_b,
        strict=strict,
        pull_if_missing=pull_if_missing,
    )
    model = resolved.to_model_spec(context_length=context_length)
    return resolved, plan_deployment(model, hardware, evaluator_config=evaluator_config)


def plan_deployment_from_gguf(
    gguf_path: str | Path,
    hardware: HardwareProfile | None = None,
    *,
    evaluator_config: EvaluatorConfig | None = None,
    context_length: int = 4096,
) -> tuple[ResolvedModel, DeploymentStrategy]:
    """
    Resolve a local GGUF artifact, then plan deployment.

    Returns:
        Tuple of resolved footprint (audit trail) and strategy.
    """
    resolved = resolve_gguf(gguf_path)
    model = resolved.to_model_spec(context_length=context_length)
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
    max_tokens: int = 256,
    ollama_base_url: str | None = None,
) -> tuple[ResolvedModel, DeploymentStrategy, InferenceResult]:
    """
    Plan deployment, then run inference with Ollama or llama-cli.

    ``runtime="auto"``: Ollama for non-.gguf refs, llama-cli for ``.gguf`` paths.

    For Ollama names, downloads the model with ``ollama pull`` when missing *before*
    resolve/plan, so blob-based planning can run.
    """
    path = Path(model_ref).expanduser()
    use_llama = runtime == "llama.cpp" or (runtime == "auto" and path.suffix.lower() == ".gguf")
    use_ollama = not use_llama

    resolved, strategy = plan_deployment_from_model(
        model_ref,
        hardware,
        evaluator_config=evaluator_config,
        expect_parameter_count_b=expect_parameter_count_b,
        strict=strict,
        pull_if_missing=pull_if_missing and use_ollama,
    )

    if use_llama:
        result = llama_cpp_generate(resolved, prompt, strategy, max_tokens=max_tokens)
        return resolved, strategy, result

    ensure_ollama_available(ollama_base_url)
    result = ollama_generate(
        model_ref,
        prompt,
        strategy,
        base_url=ollama_base_url,
    )
    return resolved, strategy, result
