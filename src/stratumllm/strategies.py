"""Deployment strategy profiles and runtime export helpers."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class DeploymentStrategyKind(str, Enum):
    """High-level placement mode returned by the evaluator."""

    FULL_GPU = "full_gpu"
    DISTRIBUTED = "distributed"
    STRATIFIED_OFFLOAD = "stratified_offload"


class StrategyRationale(BaseModel):
    """Human-readable explanation for logs and observability."""

    model_config = {"frozen": True}

    summary: str
    weight_gib: float
    vram_budget_gib: float
    ram_budget_gib: float
    notes: list[str] = Field(default_factory=list)


class FullGPUStrategy(BaseModel):
    """Full model residency on one accelerator (VRAM or unified Metal pool)."""

    model_config = {"frozen": True}

    kind: Literal[DeploymentStrategyKind.FULL_GPU] = DeploymentStrategyKind.FULL_GPU
    device_index: int = Field(ge=0)
    device_name: str
    host_memory_unified: bool = Field(
        default=False,
        description="True when all layers use a unified memory pool (e.g. Apple Metal).",
    )
    rationale: StrategyRationale

    def to_ollama_options(self) -> dict[str, Any]:
        """Options map suitable for Ollama ``/api/generate`` or Modelfile env hints."""
        return {
            "num_gpu": 999,
            "main_gpu": self.device_index,
        }

    def to_ollama_env(self) -> dict[str, str]:
        """Environment variables consumed by the Ollama server on some platforms."""
        env: dict[str, str] = {"OLLAMA_NUM_GPU": "999"}
        if not self.host_memory_unified:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device_index)
        return env

    def to_llama_cpp_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``llama_cpp.Llama`` (or llama-server CLI flags)."""
        kwargs: dict[str, Any] = {
            "n_gpu_layers": -1,
            "main_gpu": self.device_index,
        }
        if self.host_memory_unified:
            kwargs["use_mmap"] = False
        return kwargs


class DistributedStrategy(BaseModel):
    """Tensor-parallel or multi-GPU full residency across devices."""

    model_config = {"frozen": True}

    kind: Literal[DeploymentStrategyKind.DISTRIBUTED] = DeploymentStrategyKind.DISTRIBUTED
    device_indices: list[int] = Field(min_length=2)
    tensor_split: list[Annotated[float, Field(gt=0)]]
    device_names: list[str]
    rationale: StrategyRationale

    def to_ollama_options(self) -> dict[str, Any]:
        return {"num_gpu": 999}

    def to_ollama_env(self) -> dict[str, str]:
        visible = ",".join(str(i) for i in self.device_indices)
        return {
            "OLLAMA_NUM_GPU": "999",
            "CUDA_VISIBLE_DEVICES": visible,
        }

    def to_llama_cpp_kwargs(self) -> dict[str, Any]:
        return {
            "n_gpu_layers": -1,
            "tensor_split": self.tensor_split,
        }


class StratifiedOffloadStrategy(BaseModel):
    """
    Partial GPU residency with remaining layers in system RAM (strata).

    Maps cleanly to llama.cpp ``n_gpu_layers`` and Ollama partial GPU loading.
    """

    model_config = {"frozen": True}

    kind: Literal[DeploymentStrategyKind.STRATIFIED_OFFLOAD] = (
        DeploymentStrategyKind.STRATIFIED_OFFLOAD
    )
    primary_device_index: int = Field(ge=0)
    primary_device_name: str
    gpu_layers: Annotated[int, Field(ge=0)]
    total_layers: Annotated[int, Field(gt=0)]
    cpu_layers: Annotated[int, Field(ge=0)]
    gpu_weight_gib: float = Field(ge=0)
    cpu_weight_gib: float = Field(ge=0)
    use_mmap: bool = True
    host_memory_unified: bool = False
    rationale: StrategyRationale

    def to_ollama_options(self) -> dict[str, Any]:
        return {
            "num_gpu": self.gpu_layers,
            "main_gpu": self.primary_device_index,
        }

    def to_ollama_env(self) -> dict[str, str]:
        env: dict[str, str] = {"OLLAMA_NUM_GPU": str(self.gpu_layers)}
        if not self.host_memory_unified:
            env["CUDA_VISIBLE_DEVICES"] = str(self.primary_device_index)
        return env

    def to_llama_cpp_kwargs(self) -> dict[str, Any]:
        return {
            "n_gpu_layers": self.gpu_layers,
            "main_gpu": self.primary_device_index,
            "use_mmap": self.use_mmap,
        }


DeploymentStrategy = FullGPUStrategy | DistributedStrategy | StratifiedOffloadStrategy
