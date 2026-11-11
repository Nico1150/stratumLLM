#!/usr/bin/env python3
"""Example: plan deployment for a local model on this machine."""

from stratumllm import DeploymentStrategyEvaluator, ModelSpec, discover_hardware_profile
from stratumllm.strategies import (
    DistributedStrategy,
    FullGPUStrategy,
    StratifiedOffloadStrategy,
)


def main() -> None:
    hw = discover_hardware_profile()
    model = ModelSpec(name="llama3.1-8b", parameter_count_b=8.0, context_length=8192)
    strategy = DeploymentStrategyEvaluator().evaluate(hw, model)

    print(f"Hardware: {hw.total_vram_gib():.1f} GiB VRAM, {hw.system_ram_gib:.1f} GiB RAM")
    print(f"Model weights (~{model.quantization.value}): {model.estimated_weight_gib():.2f} GiB")
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


if __name__ == "__main__":
    main()
