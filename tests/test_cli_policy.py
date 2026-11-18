"""CLI evaluator policy mapping."""

from __future__ import annotations

import argparse

from stratumllm.cli_policy import add_evaluator_policy_args, build_evaluator_config


def test_build_evaluator_config_empty() -> None:
    parser = argparse.ArgumentParser()
    add_evaluator_policy_args(parser)
    args = parser.parse_args([])
    assert build_evaluator_config(args) is None


def test_build_evaluator_config_flags() -> None:
    parser = argparse.ArgumentParser()
    add_evaluator_policy_args(parser)
    args = parser.parse_args(
        ["--max-cpu-offload-gib", "12", "--uma-metal-fraction", "0.8", "--allow-cpu-only"]
    )
    cfg = build_evaluator_config(args)
    assert cfg is not None
    assert cfg.max_cpu_offload_gib == 12
    assert cfg.uma_metal_residency_fraction == 0.8
    assert cfg.allow_cpu_only is True