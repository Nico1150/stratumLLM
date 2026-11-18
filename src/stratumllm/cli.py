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
from stratumllm.cli_ollama_kv import (
    effective_check_ollama_kv,
    sync_ollama_kv_if_explicit,
    write_ollama_env_if_requested,
)
from stratumllm.cli_policy import add_evaluator_policy_args, build_evaluator_config
from stratumllm.kv_cache import (
    KvCacheStorage,
    format_ollama_kv_serve_lines,
    parse_kv_cache_storage,
    resolve_kv_cache_storage,
)
from stratumllm.plan_output import (
    build_plan_document,
    build_run_document,
    collect_ollama_kv_warnings,
    emit_plan_json,
    emit_run_json,
    parse_context_sweep,
    print_context_sweep_table,
    run_context_sweep,
)
from stratumllm.resolve import ResolvedModel
from stratumllm.runtime import RuntimeKind, infer_runtime_kind
from stratumllm.strategies import (
    DeploymentStrategy,
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


def _kv_storage_for_cli(flag: str | None, runtime: RuntimeKind) -> KvCacheStorage:
    explicit = parse_kv_cache_storage(flag) if flag else None
    return resolve_kv_cache_storage(explicit=explicit, runtime=runtime)


def _print_resolved(resolved: ResolvedModel) -> None:
    quant = resolved.quantization.value if resolved.quantization else "unknown"
    layers = resolved.num_layers if resolved.num_layers is not None else "?"
    print(f"Resolved: {resolved.name} ({resolved.source.value})")
    print(f"  weights: {resolved.weight_size_gib:.2f} GiB | layers: {layers} | quant: {quant}")
    qsrc = resolved.metadata.get("stratum.quantization_source")
    if qsrc:
        print(f"  quant source: {qsrc}")
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


def _emit_ollama_kv_notes(storage: KvCacheStorage, runtime: RuntimeKind) -> None:
    if runtime != "ollama":
        return
    for line in format_ollama_kv_serve_lines(storage):
        print(line)


def _emit_ollama_kv_warnings(lines: list[str], *, as_json: bool) -> None:
    if not lines or as_json:
        return
    for line in lines:
        print(line, file=sys.stderr)


def _print_planning_spec(spec: ModelSpec) -> None:
    print("Planning spec:")
    print(f"  context_length: {spec.context_length}")
    print(f"  kv_cache_storage: {spec.kv_cache_storage.value} ({spec.kv_cache_storage.element_bytes()} B/elem)")
    print(f"  estimated_kv_cache: {spec.estimated_kv_cache_gib():.2f} GiB")


def _finish_plan_output(
    *,
    as_json: bool,
    hw,
    kind: RuntimeKind,
    spec: ModelSpec,
    strategy: DeploymentStrategy,
    resolved: ResolvedModel | None,
    storage: KvCacheStorage,
    ollama_warnings: list[str],
    sweep_rows: list[dict] | None,
) -> None:
    if as_json:
        doc = build_plan_document(
            hardware=hw,
            runtime=kind,
            planning_spec=spec,
            strategy=strategy,
            resolved=resolved,
            kv_storage=storage,
            ollama_kv_warnings=ollama_warnings or None,
            context_sweep=sweep_rows,
        )
        emit_plan_json(doc)
        return

    if resolved is not None:
        _print_resolved(resolved)
    _print_planning_spec(spec)
    _print_strategy(strategy)
    _emit_ollama_kv_notes(storage, kind)
    _emit_ollama_kv_warnings(ollama_warnings, as_json=False)
    if sweep_rows is not None:
        print()
        print_context_sweep_table(sweep_rows)


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
        "--context-sweep",
        default=None,
        metavar="SPEC",
        help="Compare strategies across contexts: 4096,8192 or range 4096:32768:4096",
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "ollama", "llama.cpp"),
        default="auto",
        help="Runtime for KV storage assumptions (auto: .gguf → llama.cpp, else Ollama)",
    )
    parser.add_argument(
        "--kv-storage",
        default=None,
        metavar="DTYPE",
        help="K/V cache dtype for planning: f16, q8_0, q4_0 (default: OLLAMA_KV_CACHE_TYPE or f16)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON document (hardware, spec, strategy, optional sweep)",
    )
    parser.add_argument(
        "--check-ollama-kv",
        action="store_true",
        help="Verificar OLLAMA_KV_CACHE_TYPE del daemon (auto con --kv-storage)",
    )
    parser.add_argument(
        "--no-check-ollama-kv",
        action="store_true",
        help="No verificar KV del servidor aunque uses --kv-storage",
    )
    parser.add_argument(
        "--strict-ollama-kv",
        action="store_true",
        help="Fail when Ollama server KV dtype cannot be verified or does not match",
    )
    parser.add_argument(
        "--restart-ollama",
        action="store_true",
        help="Con --kv-storage: reinicia ollama serve con OLLAMA_KV_CACHE_TYPE (destructivo)",
    )
    parser.add_argument(
        "--write-ollama-env",
        default=None,
        metavar="FILE",
        help="Escribe script export (source) con OLLAMA_KV_CACHE_TYPE del plan",
    )
    add_evaluator_policy_args(parser)
    args = parser.parse_args(argv)
    eval_config = build_evaluator_config(args)

    hw = discover_hardware_profile()
    if not args.json:
        print(f"Hardware: {hw.total_vram_gib():.1f} GiB VRAM, {hw.system_ram_gib:.1f} GiB RAM")
        if hw.unified_memory:
            print(f"Unified pool budget: {hw.unified_weight_budget_gib():.1f} GiB")

    sweep_contexts: list[int] | None = None
    if args.context_sweep:
        sweep_contexts = parse_context_sweep(args.context_sweep)

    if args.model and not args.manual:
        runtime: RuntimeKind = args.runtime  # type: ignore[assignment]
        kind = infer_runtime_kind(args.model, runtime)
        kv_explicit = parse_kv_cache_storage(args.kv_storage) if args.kv_storage else None
        storage = _kv_storage_for_cli(args.kv_storage, kind)
        sync_ollama_kv_if_explicit(
            kv_flag=args.kv_storage,
            storage=storage,
            runtime=kind,
            restart_ollama=args.restart_ollama,
            as_json=args.json,
        )
        write_ollama_env_if_requested(
            path=args.write_ollama_env,
            storage=storage,
            runtime=kind,
            as_json=args.json,
        )
        resolved, strategy = plan_deployment_from_model(
            args.model,
            hw,
            evaluator_config=eval_config,
            expect_parameter_count_b=args.expect_params_b,
            strict=args.strict,
            pull_if_missing=args.pull,
            context_length=args.context,
            runtime=kind,
            kv_cache_storage=kv_explicit,
        )
        spec = resolved.to_model_spec(
            context_length=args.context,
            kv_cache_storage=storage,
        )
        ollama_warnings = collect_ollama_kv_warnings(
            storage,
            kind,
            check=effective_check_ollama_kv(args, kind),
            strict=args.strict_ollama_kv,
        )
        sweep_rows = None
        if sweep_contexts is not None:
            sweep_rows = run_context_sweep(
                hw,
                resolved=resolved,
                manual_spec=None,
                storage=storage,
                contexts=sweep_contexts,
                evaluator_config=eval_config,
            )
        _finish_plan_output(
            as_json=args.json,
            hw=hw,
            kind=kind,
            spec=spec,
            strategy=strategy,
            resolved=resolved,
            storage=storage,
            ollama_warnings=ollama_warnings,
            sweep_rows=sweep_rows,
        )
        return 0

    manual_rt: RuntimeKind = args.runtime if args.runtime != "auto" else "ollama"  # type: ignore[assignment]
    manual_storage = _kv_storage_for_cli(args.kv_storage, manual_rt)
    sync_ollama_kv_if_explicit(
        kv_flag=args.kv_storage,
        storage=manual_storage,
        runtime=manual_rt,
        restart_ollama=args.restart_ollama,
        as_json=args.json,
    )
    write_ollama_env_if_requested(
        path=args.write_ollama_env,
        storage=manual_storage,
        runtime=manual_rt,
        as_json=args.json,
    )
    model = ModelSpec(
        name=args.name,
        parameter_count_b=args.params_b,
        context_length=args.context or 8192,
        kv_cache_storage=manual_storage,
    )
    if not args.json:
        print(f"Model weights (~{model.quantization.value}): {model.estimated_weight_gib():.2f} GiB")
    strategy = DeploymentStrategyEvaluator(eval_config).evaluate(hw, model)
    ollama_warnings = collect_ollama_kv_warnings(
        manual_storage,
        manual_rt,
        check=effective_check_ollama_kv(args, manual_rt),
        strict=args.strict_ollama_kv,
    )
    sweep_rows = None
    if sweep_contexts is not None:
        sweep_rows = run_context_sweep(
            hw,
            resolved=None,
            manual_spec=model,
            storage=manual_storage,
            contexts=sweep_contexts,
            evaluator_config=eval_config,
        )
    _finish_plan_output(
        as_json=args.json,
        hw=hw,
        kind=manual_rt,
        spec=model,
        strategy=strategy,
        resolved=None,
        storage=manual_storage,
        ollama_warnings=ollama_warnings,
        sweep_rows=sweep_rows,
    )
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
    parser.add_argument(
        "--kv-storage",
        default=None,
        metavar="DTYPE",
        help="K/V cache dtype: f16, q8_0, q4_0 (llama.cpp flags; Ollama uses server env)",
    )
    parser.add_argument(
        "--check-ollama-kv",
        action="store_true",
        help="Verificar OLLAMA_KV_CACHE_TYPE del daemon (auto con --kv-storage)",
    )
    parser.add_argument(
        "--no-check-ollama-kv",
        action="store_true",
        help="No verificar KV del servidor aunque uses --kv-storage",
    )
    parser.add_argument(
        "--strict-ollama-kv",
        action="store_true",
        help="Fail run when Ollama server KV dtype cannot be verified or does not match",
    )
    parser.add_argument(
        "--write-ollama-env",
        default=None,
        metavar="FILE",
        help="Escribe script export (source) con OLLAMA_KV_CACHE_TYPE del plan",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON with plan, resolved model, strategy, and inference text",
    )
    parser.add_argument(
        "--check-ollama-ps",
        action="store_true",
        help="Tras Ollama, comparar /api/ps (VRAM/context) con el plan",
    )
    parser.add_argument(
        "--strict-ollama-ps",
        action="store_true",
        help="Falla si /api/ps no coincide con el plan",
    )
    parser.add_argument(
        "--restart-ollama",
        action="store_true",
        help="Con --kv-storage: reinicia ollama serve con OLLAMA_KV_CACHE_TYPE (destructivo)",
    )
    add_evaluator_policy_args(parser)
    args = parser.parse_args(argv)
    eval_config = build_evaluator_config(args)

    kind = infer_runtime_kind(args.model, args.runtime)  # type: ignore[arg-type]
    kv_explicit = parse_kv_cache_storage(args.kv_storage) if args.kv_storage else None
    storage = resolve_kv_cache_storage(explicit=kv_explicit, runtime=kind)
    write_ollama_env_if_requested(
        path=args.write_ollama_env,
        storage=storage,
        runtime=kind,
        as_json=args.json,
    )
    if kind == "ollama" and not args.json and not args.kv_storage:
        _emit_ollama_kv_notes(storage, kind)

    check_kv = effective_check_ollama_kv(args, kind)
    resolved, strategy, result = run_inference(
        args.model,
        args.prompt,
        evaluator_config=eval_config,
        runtime=args.runtime,  # type: ignore[arg-type]
        pull_if_missing=not args.no_pull,
        context_length=args.context,
        kv_cache_storage=kv_explicit,
        restart_ollama=args.restart_ollama and bool(args.kv_storage),
        check_ollama_kv=check_kv,
        strict_ollama_kv=args.strict_ollama_kv,
        check_ollama_ps=args.check_ollama_ps,
        strict_ollama_ps=args.strict_ollama_ps,
    )
    if args.json:
        from stratumllm.run.ollama_ps import check_ollama_ps_alignment

        spec = resolved.to_model_spec(
            context_length=args.context,
            kv_cache_storage=storage,
        )
        hw = discover_hardware_profile()
        kv_warnings = collect_ollama_kv_warnings(
            storage, kind, check=check_kv, strict=False
        )
        num_ctx = args.context if args.context is not None else (resolved.context_length or 4096)
        ps_warnings = (
            check_ollama_ps_alignment(
                args.model,
                strategy,
                planned_context=num_ctx,
                strict=False,
            )
            if kind == "ollama" and args.check_ollama_ps
            else []
        )
        emit_run_json(
            build_run_document(
                hardware=hw,
                runtime=kind,
                planning_spec=spec,
                resolved=resolved,
                strategy=strategy,
                kv_storage=storage,
                prompt=args.prompt,
                response_text=result.text,
                inference_runtime=result.runtime,
                raw_response=result.raw,
                ollama_kv_warnings=kv_warnings or None,
                ollama_ps_warnings=ps_warnings or None,
            )
        )
        return 0

    print(f"--- {result.runtime} ({strategy.kind.value}) ---")
    print(f"Model: {resolved.name} | weights: {resolved.weight_size_gib:.2f} GiB")
    print(f"KV (plan): {strategy.rationale.kv_cache_gib:.2f} GiB")
    print(result.text)
    return 0


def _entry_plan() -> None:
    sys.exit(plan_main())


def _entry_run() -> None:
    sys.exit(run_main())