#!/usr/bin/env python3
"""Plan deployment from an Ollama model name, a GGUF path, or a manual ModelSpec."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stratumllm import (
    DeploymentStrategyEvaluator,
    ModelSpec,
    discover_hardware_profile,
    plan_deployment_from_model,
)
from stratumllm.resolve import ResolvedModel
from stratumllm.strategies import (
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)


def _print_strategy(strategy: FullGPUStrategy | DistributedStrategy | StratifiedOffloadStrategy) -> None:
    print(f"Strategy: {strategy.kind.value}")
    print(f"Rationale: {strategy.rationale.summary}")

    if isinstance(strategy, FullGPUStrategy):
        print("llama.cpp:", strategy.to_llama_cpp_kwargs())
        print("Ollama env:", strategy.to_ollama_env())
    elif isinstance(strategy, DistributedStrategy):
        print("tensor_split:", strategy.tensor_split)
        print("llama.cpp:", strategy.to_llama_cpp_kwargs())
    elif isinstance(strategy, StratifiedOffloadStrategy):
        print(
            f"layers GPU/CPU: {strategy.gpu_layers}/{strategy.cpu_layers} "
            f"({strategy.gpu_weight_gib} GiB / {strategy.cpu_weight_gib} GiB)"
        )
        print("llama.cpp:", strategy.to_llama_cpp_kwargs())
        print("Ollama options:", strategy.to_ollama_options())


def _print_resolved(resolved: ResolvedModel) -> None:
    quant = resolved.quantization.value if resolved.quantization else "unknown"
    layers = resolved.num_layers if resolved.num_layers is not None else "?"
    print(f"Resolved: {resolved.name} ({resolved.source.value})")
    print(f"  weights: {resolved.weight_size_gib:.2f} GiB | layers: {layers} | quant: {quant}")
    if resolved.parameter_count_b is not None:
        print(f"  params: {resolved.parameter_count_b:.1f}B")
    if resolved.artifact_path is not None:
        print(f"  artifact: {resolved.artifact_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        help="Ollama name (qwen2.5:7b) or path to a .gguf file",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Ignore model arg; use --name and --params-b only",
    )
    parser.add_argument("--name", default="llama3.1-8b", help="Model name in manual mode")
    parser.add_argument("--params-b", type=float, default=8.0, help="Billions of parameters (manual)")
    parser.add_argument(
        "--expect-params-b",
        type=float,
        default=None,
        help="Verify resolved params match this value (±12%%)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on parameter mismatch when --expect-params-b is set",
    )
    args = parser.parse_args(argv)

    hw = discover_hardware_profile()
    print(f"Hardware: {hw.total_vram_gib():.1f} GiB VRAM, {hw.system_ram_gib:.1f} GiB RAM")
    if hw.unified_memory:
        print(f"Unified pool budget: {hw.unified_weight_budget_gib():.1f} GiB")

    if args.model and not args.manual:
        resolved, strategy = plan_deployment_from_model(
            args.model,
            hw,
            expect_parameter_count_b=args.expect_params_b,
            strict=args.strict,
        )
        _print_resolved(resolved)
        _print_strategy(strategy)
        return 0

    model = ModelSpec(name=args.name, parameter_count_b=args.params_b, context_length=8192)
    print(f"Model weights (~{model.quantization.value}): {model.estimated_weight_gib():.2f} GiB")
    strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
    _print_strategy(strategy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
