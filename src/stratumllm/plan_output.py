"""Serialize and print deployment plan results (CLI / JSON)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from stratumllm.exceptions import ConfigurationError, InsufficientResourcesError
from stratumllm.hardware import HardwareProfile
from stratumllm.kv_cache import KvCacheStorage, ollama_serve_env_hints
from stratumllm.model_spec import ModelSpec
from stratumllm.resolve import ResolvedModel
from stratumllm.run.ollama import check_ollama_kv_alignment
from stratumllm.runtime import RuntimeKind
from stratumllm.strategies import DeploymentStrategy


def parse_context_sweep(spec: str) -> list[int]:
    """
    Parse ``4096,8192`` or ``4096:32768:4096`` (start:end:step, inclusive end).
    """
    text = spec.strip()
    if not text:
        raise ConfigurationError("--context-sweep must not be empty.")
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ConfigurationError(
                "Use comma list (4096,8192) or range start:end:step (4096:32768:4096)."
            )
        start, end, step = (int(parts[0]), int(parts[1]), int(parts[2]))
        if step <= 0 or start <= 0 or end < start:
            raise ConfigurationError("Invalid context sweep range.")
        return list(range(start, end + 1, step))
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ConfigurationError("No context lengths in --context-sweep.")
    return values


def run_context_sweep(
    hardware: HardwareProfile,
    *,
    resolved: ResolvedModel | None,
    manual_spec: ModelSpec | None,
    storage: KvCacheStorage,
    contexts: list[int],
) -> list[dict[str, Any]]:
    from stratumllm import DeploymentStrategyEvaluator

    evaluator = DeploymentStrategyEvaluator()
    rows: list[dict[str, Any]] = []
    for ctx in contexts:
        if resolved is not None:
            spec = resolved.to_model_spec(context_length=ctx, kv_cache_storage=storage)
        elif manual_spec is not None:
            spec = manual_spec.model_copy(update={"context_length": ctx})
        else:
            raise ValueError("resolved or manual_spec required")
        row: dict[str, Any] = {
            "context_length": ctx,
            "kv_cache_gib": round(spec.estimated_kv_cache_gib(), 4),
        }
        try:
            strategy = evaluator.evaluate(hardware, spec)
            row["strategy"] = strategy.kind.value
            row["summary"] = strategy.rationale.summary
        except InsufficientResourcesError as exc:
            row["strategy"] = None
            row["error"] = str(exc)
        rows.append(row)
    return rows


def print_context_sweep_table(rows: list[dict[str, Any]]) -> None:
    print("Context sweep (strategy vs context_length):")
    print(f"{'context':>10}  {'KV GiB':>8}  strategy")
    print("-" * 56)
    for row in rows:
        ctx = row["context_length"]
        kv = row["kv_cache_gib"]
        if row.get("strategy"):
            print(f"{ctx:>10}  {kv:8.2f}  {row['strategy']}")
        else:
            err = row.get("error", "insufficient")
            short = err if len(err) < 40 else err[:37] + "..."
            print(f"{ctx:>10}  {kv:8.2f}  — ({short})")


def _model_json(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def build_plan_document(
    *,
    hardware: HardwareProfile,
    runtime: RuntimeKind,
    planning_spec: ModelSpec,
    strategy: DeploymentStrategy,
    resolved: ResolvedModel | None = None,
    kv_storage: KvCacheStorage,
    ollama_kv_warnings: list[str] | None = None,
    context_sweep: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "hardware": _model_json(hardware),
        "runtime": runtime,
        "planning_spec": _model_json(planning_spec),
        "strategy": _model_json(strategy),
        "kv_cache_storage": kv_storage.value,
    }
    if resolved is not None:
        doc["resolved"] = _model_json(resolved)
    if runtime == "ollama":
        doc["ollama_serve_env"] = ollama_serve_env_hints(kv_storage)
    if ollama_kv_warnings:
        doc["ollama_kv_warnings"] = ollama_kv_warnings
    if context_sweep is not None:
        doc["context_sweep"] = context_sweep
    return doc


def emit_plan_json(document: dict[str, Any]) -> None:
    print(json.dumps(document, indent=2, ensure_ascii=False))


def collect_ollama_kv_warnings(
    storage: KvCacheStorage,
    runtime: RuntimeKind,
    *,
    check: bool,
    strict: bool,
) -> list[str]:
    if runtime != "ollama" or not (check or strict):
        return []
    return check_ollama_kv_alignment(storage, strict=strict)