#!/usr/bin/env python3
"""Compare placement strategy as context length grows (library API)."""

from __future__ import annotations

from stratumllm import DeploymentStrategyEvaluator, ModelSpec, discover_hardware_profile
from stratumllm.plan_output import print_context_sweep_table, run_context_sweep

if __name__ == "__main__":
    hw = discover_hardware_profile()
    spec = ModelSpec(name="llama3.1-8b", parameter_count_b=8.0, context_length=4096)
    rows = run_context_sweep(
        hw,
        resolved=None,
        manual_spec=spec,
        storage=spec.kv_cache_storage,
        contexts=[4096, 8192, 16384, 32768, 65536],
    )
    print_context_sweep_table(rows)
    # CLI equivalent:
    #   stratumllm-plan --manual --context-sweep 4096,8192,16384,32768,65536