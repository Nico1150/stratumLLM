"""Execute inference using a planned deployment strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stratumllm.strategies import (
    DeploymentStrategy,
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Outcome of a single local generation call."""

    text: str
    model_ref: str
    runtime: str
    strategy_kind: str
    raw: dict[str, Any] | None = None


def strategy_to_ollama_options(strategy: DeploymentStrategy) -> dict[str, Any]:
    """Map a deployment strategy to Ollama ``options`` for ``/api/generate``."""
    if isinstance(strategy, FullGPUStrategy):
        return strategy.to_ollama_options()
    if isinstance(strategy, DistributedStrategy):
        return strategy.to_ollama_options()
    if isinstance(strategy, StratifiedOffloadStrategy):
        return strategy.to_ollama_options()
    raise TypeError(f"Unknown strategy type: {type(strategy)!r}")


def strategy_to_llama_cpp_cli_args(strategy: DeploymentStrategy) -> list[str]:
    """Extra CLI flags for ``llama-cli`` (after ``-m`` model path)."""
    kwargs = (
        strategy.to_llama_cpp_kwargs()
        if isinstance(strategy, (FullGPUStrategy, DistributedStrategy, StratifiedOffloadStrategy))
        else {}
    )
    args: list[str] = []
    n_gpu = kwargs.get("n_gpu_layers", -1)
    args.extend(["-ngl", str(n_gpu)])
    if kwargs.get("use_mmap") is False:
        args.extend(["--no-mmap"])
    return args
