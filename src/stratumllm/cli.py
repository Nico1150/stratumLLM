"""Console entry points for plan and run."""

from __future__ import annotations

import argparse
import sys

from stratumllm import (
    DeploymentStrategyEvaluator,
    ModelSpec,
    discover_hardware_profile,
    plan_deployment_from_model,
    run_inference,
)
from stratumllm.kv_cache import kv_storage_for_runtime
from stratumllm.resolve import ResolvedModel
from stratumllm.runtime import RuntimeKind, infer_runtime_kind
from stratumllm.strategies import (
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)


def _print_strategy(strategy: FullGPUStrategy | DistributedStrategy | StratifiedOffloadStrategy) -> None:
    r = strategy.rationale
    print(f"Strategy: {strategy.kind.value}")
    print(f"Rationale: {r.summary}")
    print(f"  weights (plan): {r.weight_gib:.2f} GiB | KV (plan): {r.kv_cache_gib:.2f} GiB")
    print(f"  budgets: VRAM/unified {r.vram_budget_gib:.2f} GiB, RAM {r.ram_budget_gib:.2f} GiB")

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
    if resolved.context_length is not None:
        print(f"  context (artifact): {resolved.context_length}")
    attn: list[str] = []
    if resolved.hidden_size is not None:
        attn.append(f"hidden={resolved.hidden_size}")
    if resolved.num_kv_heads is not None:
        attn.append(f"kv_heads={resolved.num_kv_heads}")
    if resolved.head_dim is not None:
        attn.append(f"head_dim={resolved.head_dim}")
    if attn:
        print(f"  attention: {', '.join(attn)}")
    if resolved.artifact_path is not None:
        print(f"  artifact: {resolved.artifact_path}")


def _print_planning_spec(spec: ModelSpec) -> None:
    print("Planning spec:")
    print(f"  context_length: {spec.context_length}")
    print(f"  kv_cache_storage: {spec.kv_cache_storage.value} ({spec.kv_cache_storage.element_bytes()} B/elem)")
    print(f"  estimated_kv_cache: {spec.estimated_kv_cache_gib():.2f} GiB")


def plan_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan deployment (Ollama, GGUF, or manual ModelSpec).")
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
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Run `ollama pull` when the Ollama model is not local",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override context length for KV/planning (default: artifact or 4096)",
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "ollama", "llama.cpp"),
        default="auto",
        help="Runtime for KV storage assumptions (auto: .gguf → llama.cpp, else Ollama)",
    )
    args = parser.parse_args(argv)

    hw = discover_hardware_profile()
    print(f"Hardware: {hw.total_vram_gib():.1f} GiB VRAM, {hw.system_ram_gib:.1f} GiB RAM")
    if hw.unified_memory:
        print(f"Unified pool budget: {hw.unified_weight_budget_gib():.1f} GiB")

    if args.model and not args.manual:
        runtime: RuntimeKind = args.runtime  # type: ignore[assignment]
        kind = infer_runtime_kind(args.model, runtime)
        resolved, strategy = plan_deployment_from_model(
            args.model,
            hw,
            expect_parameter_count_b=args.expect_params_b,
            strict=args.strict,
            pull_if_missing=args.pull,
            context_length=args.context,
            runtime=kind,
        )
        spec = resolved.to_model_spec(
            context_length=args.context,
            kv_cache_storage=kv_storage_for_runtime(kind),
        )
        _print_resolved(resolved)
        _print_planning_spec(spec)
        _print_strategy(strategy)
        return 0

    model = ModelSpec(name=args.name, parameter_count_b=args.params_b, context_length=args.context or 8192)
    print(f"Model weights (~{model.quantization.value}): {model.estimated_weight_gib():.2f} GiB")
    _print_planning_spec(model)
    strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
    _print_strategy(strategy)
    return 0


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan deployment and run a single prompt.")
    parser.add_argument("model", help="Ollama model name or path to .gguf")
    parser.add_argument("prompt", nargs="?", default="Why is the sky blue? Answer in one sentence.")
    parser.add_argument(
        "--runtime",
        choices=("auto", "ollama", "llama.cpp"),
        default="auto",
        help="auto: .gguf → llama-cli, else Ollama",
    )
    parser.add_argument("--no-pull", action="store_true", help="Do not run ollama pull if missing")
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Override context length for planning and num_ctx",
    )
    args = parser.parse_args(argv)

    resolved, strategy, result = run_inference(
        args.model,
        args.prompt,
        runtime=args.runtime,  # type: ignore[arg-type]
        pull_if_missing=not args.no_pull,
        context_length=args.context,
    )
    print(f"--- {result.runtime} ({strategy.kind.value}) ---")
    print(f"Model: {resolved.name} | weights: {resolved.weight_size_gib:.2f} GiB")
    print(f"KV (plan): {strategy.rationale.kv_cache_gib:.2f} GiB")
    print(result.text)
    return 0


def _entry_plan() -> None:
    sys.exit(plan_main())


def _entry_run() -> None:
    sys.exit(run_main())