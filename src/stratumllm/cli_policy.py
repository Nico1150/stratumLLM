"""Map CLI flags to ``EvaluatorConfig``."""

from __future__ import annotations

import argparse

from stratumllm.evaluator import EvaluatorConfig


def add_evaluator_policy_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("política del evaluator (adaptable por máquina)")
    group.add_argument(
        "--max-cpu-offload-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="Tope de peso en CPU/RAM en offload estratificado",
    )
    group.add_argument(
        "--max-cpu-offload-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="Fracción máxima del modelo en CPU (0–1)",
    )
    group.add_argument(
        "--max-cpu-only-weight-gib",
        type=float,
        default=None,
        metavar="GIB",
        help="Límite de peso si solo hay CPU",
    )
    group.add_argument(
        "--uma-metal-fraction",
        type=float,
        default=None,
        metavar="FRAC",
        help="Fracción del pool unified que intenta residir en Metal (Apple)",
    )
    group.add_argument(
        "--force-stratified-on-uma",
        action="store_true",
        help="Forzar offload estratificado en memoria unificada",
    )
    group.add_argument(
        "--allow-cpu-only",
        action="store_true",
        help="Permitir despliegue solo en CPU",
    )


def build_evaluator_config(args: argparse.Namespace) -> EvaluatorConfig | None:
    """Build config only when the user set at least one policy flag."""
    fields: dict[str, float | bool] = {}
    if args.max_cpu_offload_gib is not None:
        fields["max_cpu_offload_gib"] = args.max_cpu_offload_gib
    if args.max_cpu_offload_fraction is not None:
        fields["max_cpu_offload_fraction"] = args.max_cpu_offload_fraction
    if args.max_cpu_only_weight_gib is not None:
        fields["max_cpu_only_weight_gib"] = args.max_cpu_only_weight_gib
    if args.uma_metal_fraction is not None:
        fields["uma_metal_residency_fraction"] = args.uma_metal_fraction
    if getattr(args, "force_stratified_on_uma", False):
        fields["force_stratified_on_uma"] = True
    if getattr(args, "allow_cpu_only", False):
        fields["allow_cpu_only"] = True
    if not fields:
        return None
    return EvaluatorConfig(**fields)