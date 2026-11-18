"""Ollama ``/api/ps`` helpers for post-run sanity checks."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from stratumllm.exceptions import ConfigurationError
from stratumllm.run.ollama import ollama_base_url
from stratumllm.strategies import DeploymentStrategy, StratifiedOffloadStrategy


@dataclass(frozen=True, slots=True)
class OllamaRunningModel:
    name: str
    size_gib: float
    size_vram_gib: float | None
    context_length: int | None


def fetch_ollama_running_models(base_url: str | None = None) -> list[OllamaRunningModel]:
    base = (base_url or ollama_base_url()).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise ConfigurationError(f"Failed to read Ollama /api/ps: {exc}") from exc
    rows: list[OllamaRunningModel] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "")
        size_b = item.get("size")
        vram_b = item.get("size_vram")
        ctx = item.get("context_length")
        rows.append(
            OllamaRunningModel(
                name=name,
                size_gib=float(size_b) / (1024**3) if isinstance(size_b, (int, float)) else 0.0,
                size_vram_gib=float(vram_b) / (1024**3) if isinstance(vram_b, (int, float)) else None,
                context_length=int(ctx) if isinstance(ctx, (int, float)) and ctx > 0 else None,
            )
        )
    return rows


def planned_ollama_resident_gib(strategy: DeploymentStrategy) -> float:
    """Heuristic resident footprint (weights on GPU + KV) for comparison with ``/api/ps``."""
    rationale = strategy.rationale
    if isinstance(strategy, StratifiedOffloadStrategy):
        return strategy.gpu_weight_gib + rationale.kv_cache_gib
    return rationale.weight_gib + rationale.kv_cache_gib


def _match_running_model(model_ref: str, running: list[OllamaRunningModel]) -> OllamaRunningModel | None:
    ref = model_ref.strip().lower()
    for item in running:
        name = item.name.lower()
        if name == ref or name.startswith(ref) or ref.startswith(name):
            return item
    return None


def check_ollama_ps_alignment(
    model_ref: str,
    strategy: DeploymentStrategy,
    *,
    planned_context: int,
    base_url: str | None = None,
    vram_tolerance: float = 0.45,
    strict: bool = False,
) -> list[str]:
    """
    Compare ``/api/ps`` after a run with the planning heuristic.

    Uses ``size_vram`` when present, else ``size``. Context length is compared when reported.
    """
    try:
        running = fetch_ollama_running_models(base_url)
    except ConfigurationError as exc:
        msg = str(exc)
        if strict:
            raise
        return [f"warning: {msg}"]

    match = _match_running_model(model_ref, running)
    if match is None:
        msg = (
            f"Model {model_ref!r} not listed in Ollama /api/ps after run "
            "(may have unloaded already)."
        )
        if strict:
            raise ConfigurationError(msg)
        return [f"warning: {msg}"]

    planned_gib = planned_ollama_resident_gib(strategy)
    observed_gib = match.size_vram_gib if match.size_vram_gib is not None else match.size_gib
    messages: list[str] = []

    if planned_gib > 0:
        delta = abs(observed_gib - planned_gib) / planned_gib
        if delta > vram_tolerance:
            msg = (
                f"Ollama /api/ps resident ~{observed_gib:.2f} GiB vs plan ~{planned_gib:.2f} GiB "
                f"(Δ {delta:.0%}, tolerance {vram_tolerance:.0%})."
            )
            if strict:
                raise ConfigurationError(msg)
            messages.append(f"warning: {msg}")

    if match.context_length is not None and match.context_length != planned_context:
        msg = (
            f"Ollama /api/ps context_length={match.context_length} "
            f"vs planned num_ctx={planned_context}."
        )
        if strict:
            raise ConfigurationError(msg)
        messages.append(f"warning: {msg}")

    return messages