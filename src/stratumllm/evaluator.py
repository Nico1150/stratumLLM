"""Select a deployment strategy from hardware capacity and model metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass

from stratumllm.exceptions import InsufficientResourcesError
from stratumllm.hardware import GpuDevice, HardwareProfile
from stratumllm.model_spec import ModelSpec
from stratumllm.strategies import (
    DeploymentStrategy,
    DistributedStrategy,
    FullGPUStrategy,
    StrategyRationale,
    StratifiedOffloadStrategy,
)


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """Tunable thresholds for strategy selection."""

    min_gpus_for_distributed: int = 2
    distributed_min_params_b: float = 30.0
    allow_cpu_only: bool = False
    uma_metal_residency_fraction: float | None = None
    force_stratified_on_uma: bool = False
    max_cpu_offload_gib: float | None = None
    max_cpu_offload_fraction: float | None = None
    max_cpu_only_weight_gib: float | None = None


class DeploymentStrategyEvaluator:
    """
    Evaluates available hardware and returns a concrete deployment profile.

    **Discrete GPU:** Full → Distributed → Stratified (VRAM + system RAM).

    **Unified memory (UMA):** feasibility uses a single pool. If weights fit the pool
    and ``uma_metal_residency_fraction`` is 1.0 (default), prefer full-metal placement.
    Partial offload on UMA is opt-in (fraction < 1 or ``force_stratified_on_uma``).
    """

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        self._config = config or EvaluatorConfig()

    def _policy_cpu_weight_cap(self, weight_gib: float) -> float | None:
        """Optional policy ceiling on host/off-accelerator weight (GiB)."""
        caps: list[float] = []
        if self._config.max_cpu_offload_gib is not None:
            caps.append(self._config.max_cpu_offload_gib)
        if self._config.max_cpu_offload_fraction is not None:
            caps.append(weight_gib * self._config.max_cpu_offload_fraction)
        if not caps:
            return None
        return min(caps)

    def _effective_cpu_weight_cap(
        self,
        weight_gib: float,
        *,
        ram_budget: float,
        discrete_split: bool,
    ) -> float:
        """Feasibility (RAM) ∩ optional policy caps on CPU-side weights."""
        feasibility = ram_budget if discrete_split else float("inf")
        policy = self._policy_cpu_weight_cap(weight_gib)
        if policy is None:
            return feasibility
        return min(feasibility, policy)

    def evaluate(self, hardware: HardwareProfile, model: ModelSpec) -> DeploymentStrategy:
        """
        Compute the best strategy for ``model`` on ``hardware``.

        Raises:
            InsufficientResourcesError: If the model cannot be placed under constraints.
        """
        weight_gib = model.estimated_weight_gib()
        if hardware.unified_memory:
            return self._evaluate_unified(hardware, model, weight_gib)
        return self._evaluate_discrete(hardware, model, weight_gib)

    def _evaluate_discrete(
        self, hardware: HardwareProfile, model: ModelSpec, weight_gib: float
    ) -> DeploymentStrategy:
        vram_total_budget = hardware.vram_budget_gib()
        ram_budget = hardware.ram_budget_gib()

        if not hardware.gpus:
            return self._cpu_only_or_fail(hardware, model, weight_gib, ram_budget)

        best_single = self._best_single_gpu(hardware, weight_gib)
        if best_single is not None:
            gpu, budget = best_single
            return self._full_gpu(
                hardware,
                gpu,
                weight_gib,
                budget,
                ram_budget,
                summary="Model fits entirely in single-GPU VRAM budget.",
            )

        if (
            len(hardware.gpus) >= self._config.min_gpus_for_distributed
            and model.parameter_count_b >= self._config.distributed_min_params_b
            and weight_gib <= vram_total_budget
        ):
            return self._build_distributed(hardware, model, weight_gib, vram_total_budget, ram_budget)

        offload = self._build_stratified_discrete(hardware, model, weight_gib, ram_budget)
        if offload is not None:
            return offload

        raise InsufficientResourcesError(
            f"Cannot place {model.name} (~{weight_gib:.2f} GiB weights): "
            f"VRAM budget {vram_total_budget:.2f} GiB, RAM budget {ram_budget:.2f} GiB."
        )

    def _evaluate_unified(
        self, hardware: HardwareProfile, model: ModelSpec, weight_gib: float
    ) -> DeploymentStrategy:
        ram_budget = hardware.ram_budget_gib()
        pool_budget = hardware.unified_weight_budget_gib()

        if not hardware.gpus:
            return self._cpu_only_or_fail(hardware, model, weight_gib, pool_budget)

        if weight_gib > pool_budget:
            raise InsufficientResourcesError(
                f"Cannot place {model.name} (~{weight_gib:.2f} GiB weights) in unified pool "
                f"(budget {pool_budget:.2f} GiB)."
            )

        metal_fraction = (
            self._config.uma_metal_residency_fraction
            if self._config.uma_metal_residency_fraction is not None
            else hardware.uma_metal_residency_fraction
        )
        metal_weight_budget = pool_budget * metal_fraction

        prefer_full_metal = (
            not self._config.force_stratified_on_uma
            and weight_gib <= metal_weight_budget
        )
        if prefer_full_metal:
            primary = self._primary_gpu(hardware)
            return self._full_gpu(
                hardware,
                primary,
                weight_gib,
                pool_budget,
                ram_budget,
                summary="Model fits unified memory pool; full Metal residency.",
                host_memory_unified=True,
            )

        offload = self._build_stratified_unified(
            hardware,
            model,
            weight_gib,
            metal_weight_budget,
            pool_budget,
        )
        if offload is not None:
            return offload

        raise InsufficientResourcesError(
            f"Cannot place {model.name} (~{weight_gib:.2f} GiB weights) with UMA stratified split "
            f"(pool {pool_budget:.2f} GiB, Metal soft cap {metal_weight_budget:.2f} GiB)."
        )

    @staticmethod
    def _primary_gpu(hardware: HardwareProfile) -> GpuDevice:
        return max(hardware.gpus, key=lambda g: g.effective_vram_gib())

    def _full_gpu(
        self,
        hardware: HardwareProfile,
        gpu: GpuDevice,
        weight_gib: float,
        budget_gib: float,
        ram_budget: float,
        *,
        summary: str,
        host_memory_unified: bool = False,
    ) -> FullGPUStrategy:
        notes = [f"Selected GPU {gpu.index} ({gpu.name})."]
        if host_memory_unified:
            notes.append("Unified memory pool; all layers on Metal.")
        return FullGPUStrategy(
            device_index=gpu.index,
            device_name=gpu.name,
            host_memory_unified=host_memory_unified,
            rationale=self._rationale(
                summary=summary,
                weight_gib=weight_gib,
                vram_budget_gib=budget_gib,
                ram_budget_gib=ram_budget,
                notes=notes,
            ),
        )

    def _best_single_gpu(
        self, hardware: HardwareProfile, weight_gib: float
    ) -> tuple[GpuDevice, float] | None:
        candidates: list[tuple[GpuDevice, float]] = []
        for gpu in hardware.gpus:
            budget = hardware.vram_budget_gib(device_index=gpu.index)
            if weight_gib <= budget:
                candidates.append((gpu, budget))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1])
        return candidates[0]

    def _build_distributed(
        self,
        hardware: HardwareProfile,
        model: ModelSpec,
        weight_gib: float,
        vram_total_budget: float,
        ram_budget: float,
    ) -> DistributedStrategy:
        budgets = [hardware.vram_budget_gib(device_index=g.index) for g in hardware.gpus]
        total = sum(budgets)
        if total <= 0:
            raise InsufficientResourcesError("No usable VRAM budget on any GPU.")

        tensor_split = [b / total for b in budgets]
        indices = [g.index for g in hardware.gpus]
        names = [g.name for g in hardware.gpus]

        return DistributedStrategy(
            device_indices=indices,
            tensor_split=tensor_split,
            device_names=names,
            rationale=self._rationale(
                summary="Model spans multiple GPUs with proportional tensor split.",
                weight_gib=weight_gib,
                vram_budget_gib=vram_total_budget,
                ram_budget_gib=ram_budget,
                notes=[
                    f"Layer/weight split fractions: {[round(x, 3) for x in tensor_split]}",
                    "Requires runtime support for multi-GPU tensor parallelism.",
                ],
            ),
        )

    def _build_stratified_discrete(
        self,
        hardware: HardwareProfile,
        model: ModelSpec,
        weight_gib: float,
        ram_budget: float,
    ) -> StratifiedOffloadStrategy | None:
        primary = self._primary_gpu(hardware)
        vram_budget = hardware.vram_budget_gib(device_index=primary.index)
        return self._build_stratified_layers(
            hardware=hardware,
            model=model,
            weight_gib=weight_gib,
            accelerator_weight_budget=vram_budget,
            pool_budget=weight_gib,
            ram_budget=ram_budget,
            use_mmap=True,
            discrete_split=True,
        )

    def _build_stratified_unified(
        self,
        hardware: HardwareProfile,
        model: ModelSpec,
        weight_gib: float,
        metal_weight_budget: float,
        pool_budget: float,
    ) -> StratifiedOffloadStrategy | None:
        primary = self._primary_gpu(hardware)
        return self._build_stratified_layers(
            hardware=hardware,
            model=model,
            weight_gib=weight_gib,
            accelerator_weight_budget=metal_weight_budget,
            pool_budget=pool_budget,
            ram_budget=hardware.ram_budget_gib(),
            use_mmap=False,
            discrete_split=False,
            primary=primary,
        )

    def _build_stratified_layers(
        self,
        *,
        hardware: HardwareProfile,
        model: ModelSpec,
        weight_gib: float,
        accelerator_weight_budget: float,
        pool_budget: float,
        ram_budget: float,
        use_mmap: bool,
        discrete_split: bool,
        primary: GpuDevice | None = None,
    ) -> StratifiedOffloadStrategy | None:
        total_layers = model.resolved_num_layers()
        primary = primary or self._primary_gpu(hardware)
        bytes_per_layer = model.bytes_per_layer_gib()

        if accelerator_weight_budget <= 0:
            return None

        max_gpu_layers_from_accel = (
            int(accelerator_weight_budget // bytes_per_layer) if bytes_per_layer > 0 else 0
        )
        max_gpu_layers_from_accel = min(total_layers, max_gpu_layers_from_accel)
        if max_gpu_layers_from_accel <= 0:
            return None

        if weight_gib > pool_budget:
            return None

        cpu_cap = self._effective_cpu_weight_cap(
            weight_gib,
            ram_budget=ram_budget,
            discrete_split=discrete_split,
        )
        min_gpu_layers = 0
        if bytes_per_layer > 0 and cpu_cap < weight_gib:
            min_gpu_layers = math.ceil((weight_gib - cpu_cap) / bytes_per_layer)
        min_gpu_layers = min(total_layers, max(0, min_gpu_layers))

        if max_gpu_layers_from_accel < min_gpu_layers:
            return None

        max_gpu_layers = max_gpu_layers_from_accel

        cpu_weight = weight_gib - (max_gpu_layers * bytes_per_layer)
        cpu_weight = max(0.0, cpu_weight)

        cpu_layers = total_layers - max_gpu_layers
        gpu_weight = weight_gib - cpu_weight

        summary = (
            "Stratified placement on discrete GPU + system RAM."
            if discrete_split
            else "UMA policy: partial Metal residency within unified pool."
        )

        return StratifiedOffloadStrategy(
            primary_device_index=primary.index,
            primary_device_name=primary.name,
            gpu_layers=max_gpu_layers,
            total_layers=total_layers,
            cpu_layers=cpu_layers,
            gpu_weight_gib=round(gpu_weight, 3),
            cpu_weight_gib=round(cpu_weight, 3),
            use_mmap=use_mmap,
            host_memory_unified=hardware.unified_memory,
            rationale=self._rationale(
                summary=summary,
                weight_gib=weight_gib,
                vram_budget_gib=accelerator_weight_budget,
                ram_budget_gib=ram_budget,
                notes=[
                    f"{max_gpu_layers}/{total_layers} layers on GPU ({primary.name}).",
                    f"~{cpu_weight:.2f} GiB weights off accelerator path.",
                ],
            ),
        )

    def _cpu_only_or_fail(
        self,
        hardware: HardwareProfile,
        model: ModelSpec,
        weight_gib: float,
        ram_budget: float,
    ) -> DeploymentStrategy:
        total_layers = model.resolved_num_layers()
        if not self._config.allow_cpu_only:
            raise InsufficientResourcesError(
                "No GPUs detected and CPU-only placement is disabled."
            )
        cpu_cap = min(
            ram_budget,
            self._config.max_cpu_only_weight_gib
            if self._config.max_cpu_only_weight_gib is not None
            else float("inf"),
        )
        if weight_gib > cpu_cap:
            raise InsufficientResourcesError(
                "CPU-only placement exceeds RAM or policy weight cap."
            )
        return StratifiedOffloadStrategy(
            primary_device_index=0,
            primary_device_name="cpu",
            gpu_layers=0,
            total_layers=total_layers,
            cpu_layers=total_layers,
            gpu_weight_gib=0.0,
            cpu_weight_gib=weight_gib,
            use_mmap=True,
            host_memory_unified=hardware.unified_memory,
            rationale=self._rationale(
                summary="CPU-only inference (no GPU).",
                weight_gib=weight_gib,
                vram_budget_gib=0.0,
                ram_budget_gib=ram_budget,
                notes=["Enable only for smoke tests; expect low throughput."],
            ),
        )

    @staticmethod
    def _rationale(
        *,
        summary: str,
        weight_gib: float,
        vram_budget_gib: float,
        ram_budget_gib: float,
        notes: list[str],
    ) -> StrategyRationale:
        return StrategyRationale(
            summary=summary,
            weight_gib=round(weight_gib, 3),
            vram_budget_gib=round(vram_budget_gib, 3),
            ram_budget_gib=round(ram_budget_gib, 3),
            notes=notes,
        )
