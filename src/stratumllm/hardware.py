"""Pydantic schemas for hardware capacity and placement constraints."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from stratumllm.exceptions import ConfigurationError


class AcceleratorKind(str, Enum):
    """Primary compute backend for weight placement."""

    CUDA = "cuda"
    METAL = "metal"
    CPU = "cpu"
    UNKNOWN = "unknown"


class GpuDevice(BaseModel):
    """A single GPU or accelerator with dedicated memory."""

    model_config = {"frozen": True}

    index: int = Field(ge=0, description="Ordinal device index for the runtime.")
    name: str = Field(min_length=1, description="Human-readable device name.")
    vram_gib: Annotated[float, Field(gt=0, description="Total VRAM in gibibytes.")]
    vram_free_gib: Annotated[
        float | None,
        Field(default=None, gt=0, description="Free VRAM if known; else use total."),
    ] = None

    def effective_vram_gib(self) -> float:
        """VRAM budget for placement (prefer reported free memory)."""
        return self.vram_free_gib if self.vram_free_gib is not None else self.vram_gib


class HardwareProfile(BaseModel):
    """
    Validated snapshot of host resources used for strategy selection.

    All memory fields are in gibibytes (GiB). Estimates assume inference (not training).

    **Unified memory (UMA):** use ``unified_weight_budget_gib`` for feasibility and
    ``metal_residency_weight_budget_gib`` only when deliberately splitting layers.

    **Discrete GPUs:** use ``vram_budget_gib`` / ``ram_budget_gib`` as separate pools.
    """

    model_config = {"frozen": True}

    system_ram_gib: Annotated[float, Field(gt=0, description="Total system RAM.")]
    system_ram_available_gib: Annotated[
        float | None,
        Field(default=None, gt=0, description="Available RAM; defaults to a fraction of total."),
    ] = None
    gpus: list[GpuDevice] = Field(default_factory=list)
    accelerator: AcceleratorKind = Field(
        default=AcceleratorKind.UNKNOWN,
        description="Primary backend (CUDA, Metal, etc.).",
    )
    gpu_utilization_target: Annotated[
        float,
        Field(
            default=0.90,
            gt=0,
            le=1.0,
            description="Max fraction of VRAM to allocate to weights + KV budget.",
        ),
    ] = 0.90
    ram_utilization_target: Annotated[
        float,
        Field(
            default=0.75,
            gt=0,
            le=1.0,
            description="Max fraction of available RAM for CPU-offloaded weights.",
        ),
    ] = 0.75
    kv_cache_reserve_gib: Annotated[
        float,
        Field(
            default=2.0,
            ge=0,
            description="Headroom reserved per active GPU for KV cache and runtime overhead.",
        ),
    ] = 2.0
    unified_memory: bool = Field(
        default=False,
        description="True when GPU and CPU share one physical memory pool (UMA / Metal).",
    )
    uma_metal_residency_fraction: Annotated[
        float,
        Field(
            default=1.0,
            gt=0,
            le=1.0,
            description=(
                "UMA only: soft cap on fraction of unified weight budget preferred on Metal. "
                "1.0 = full-metal placement when the model fits the pool."
            ),
        ),
    ] = 1.0

    @field_validator("gpus")
    @classmethod
    def unique_gpu_indices(cls, gpus: list[GpuDevice]) -> list[GpuDevice]:
        indices = [g.index for g in gpus]
        if len(indices) != len(set(indices)):
            raise ValueError("GPU device indices must be unique.")
        return gpus

    def resolved_ram_available_gib(self) -> float:
        """Effective available RAM for offload budgeting."""
        if self.system_ram_available_gib is not None:
            return self.system_ram_available_gib
        return self.system_ram_gib * 0.65

    @model_validator(mode="after")
    def validate_ram_available(self) -> HardwareProfile:
        available = self.system_ram_available_gib
        if available is not None and available > self.system_ram_gib:
            raise ConfigurationError("Available RAM cannot exceed total system RAM.")
        return self

    def total_vram_gib(self) -> float:
        """Sum of effective VRAM across all GPUs."""
        return sum(g.effective_vram_gib() for g in self.gpus)

    def vram_budget_gib(self, device_index: int | None = None) -> float:
        """
        Usable dedicated VRAM on discrete GPU hosts (not used for UMA feasibility).

        For multi-GPU planning, ``device_index`` selects a single card; ``None`` sums budgets.
        """
        if not self.gpus:
            return 0.0

        devices = [g for g in self.gpus if device_index is None or g.index == device_index]
        if device_index is not None and not devices:
            raise ConfigurationError(f"No GPU with index {device_index}.")

        budgets: list[float] = []
        for gpu in devices:
            raw = gpu.effective_vram_gib() - self.kv_cache_reserve_gib
            budgets.append(max(0.0, raw) * self.gpu_utilization_target)
        return sum(budgets)

    def ram_budget_gib(self) -> float:
        """Usable system RAM for CPU-offloaded weights (discrete GPU hosts)."""
        return self.resolved_ram_available_gib() * self.ram_utilization_target

    def unified_weight_budget_gib(self) -> float:
        """Hard cap: total model weights that may reside in a unified memory pool."""
        return max(
            0.0,
            (self.system_ram_gib - self.kv_cache_reserve_gib) * self.ram_utilization_target,
        )

    def metal_residency_weight_budget_gib(self) -> float:
        """
        UMA: soft weight budget for Metal layer residency (stratified planning only).

        When equal to the full unified budget and the model fits the pool, the planner
        should prefer full-metal placement instead of partial offload.
        """
        if not self.unified_memory:
            raise ConfigurationError("metal_residency_weight_budget_gib applies only to UMA profiles.")
        return self.unified_weight_budget_gib() * self.uma_metal_residency_fraction
