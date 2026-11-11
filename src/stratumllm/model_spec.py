"""Model metadata and weight-size estimation for placement decisions."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class QuantizationKind(str, Enum):
    """Supported weight precisions for size estimation."""

    Q4_K_M = "q4_k_m"
    Q4_0 = "q4_0"
    Q5_K_M = "q5_k_m"
    Q8_0 = "q8_0"
    F16 = "f16"


# Approximate bytes per parameter by quantization (including modest metadata overhead).
_BYTES_PER_PARAM: dict[QuantizationKind, float] = {
    QuantizationKind.Q4_K_M: 0.56,
    QuantizationKind.Q4_0: 0.52,
    QuantizationKind.Q5_K_M: 0.64,
    QuantizationKind.Q8_0: 1.02,
    QuantizationKind.F16: 2.05,
}


class ModelSpec(BaseModel):
    """
    Describes an LLM workload for strategy selection.

    ``parameter_count_b`` is the active parameter count in billions (e.g. 7, 14, 70).
    Weight size can be supplied explicitly or derived from quantization.
    """

    model_config = {"frozen": True}

    name: str = Field(min_length=1, description="Model identifier (e.g. llama3.1-8b).")
    parameter_count_b: Annotated[float, Field(gt=0, description="Billions of parameters.")]
    quantization: QuantizationKind = Field(
        default=QuantizationKind.Q4_K_M,
        description="Default 4-bit preset for local deployment planning.",
    )
    weight_size_gib: Annotated[
        float | None,
        Field(default=None, gt=0, description="Override computed weight footprint."),
    ] = None
    num_layers: Annotated[
        int | None,
        Field(default=None, gt=0, description="Transformer blocks; inferred if omitted."),
    ] = None
    context_length: Annotated[int, Field(default=4096, ge=256)] = 4096
    architecture: str = Field(default="transformer", description="Family hint for heuristics.")

    def resolved_num_layers(self) -> int:
        """Layer count used for stratified offload (explicit or family heuristic)."""
        if self.num_layers is not None:
            return self.num_layers
        params = self.parameter_count_b
        if params <= 4 or params <= 9:
            return 32
        if params <= 15:
            return 40
        if params <= 35:
            return 60
        if params <= 75:
            return 80
        return max(80, int(params * 1.1))

    def estimated_weight_gib(self) -> float:
        """Return model weight size in GiB (1024^3 bytes)."""
        if self.weight_size_gib is not None:
            return self.weight_size_gib
        params = self.parameter_count_b * 1e9
        bytes_per_param = _BYTES_PER_PARAM[self.quantization]
        gib = (params * bytes_per_param) / (1024**3)
        # Small fixed overhead for embeddings / output head in GGUF bundles.
        return gib * 1.05

    def bytes_per_layer_gib(self) -> float:
        """Average weight bytes per transformer layer (uniform split heuristic)."""
        layers = self.resolved_num_layers()
        return self.estimated_weight_gib() / layers
