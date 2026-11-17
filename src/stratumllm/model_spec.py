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
def bytes_per_parameter(quantization: QuantizationKind) -> float:
    """Planning estimate of bytes per parameter for a quantization preset."""
    return _BYTES_PER_PARAM[quantization]


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
    hidden_size: Annotated[
        int | None,
        Field(default=None, gt=0, description="Model width; used for KV cache sizing."),
    ] = None
    num_kv_heads: Annotated[
        int | None,
        Field(default=None, gt=0, description="GQA KV heads; inferred when omitted."),
    ] = None
    head_dim: Annotated[
        int | None,
        Field(default=None, gt=0, description="Attention head dimension; default 128."),
    ] = None

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

    def estimated_kv_cache_gib(self, *, kv_element_bytes: float = 2.0) -> float:
        """
        Estimate KV cache for ``context_length`` tokens (batch size 1, fp16 by default).

        Uses ``hidden_size`` / GQA fields when set; otherwise family heuristics from
        ``parameter_count_b``.
        """
        layers = self.resolved_num_layers()
        hidden = self.hidden_size if self.hidden_size is not None else _heuristic_hidden_size(
            self.parameter_count_b
        )
        head_dim = self.head_dim if self.head_dim is not None else 128
        num_heads = max(1, hidden // head_dim)
        if self.num_kv_heads is not None:
            n_kv = self.num_kv_heads
        else:
            n_kv = _heuristic_num_kv_heads(self.parameter_count_b, num_heads)
        n_kv = min(n_kv, num_heads)
        bytes_total = 2 * layers * n_kv * head_dim * self.context_length * kv_element_bytes
        return bytes_total / (1024**3)


def _heuristic_hidden_size(parameter_count_b: float) -> int:
    if parameter_count_b <= 3.5:
        return 3072
    if parameter_count_b <= 9:
        return 4096
    if parameter_count_b <= 15:
        return 5120
    if parameter_count_b <= 80:
        return 8192
    return 12288


def _heuristic_num_kv_heads(parameter_count_b: float, num_heads: int) -> int:
    if parameter_count_b <= 35:
        return min(8, num_heads)
    return min(8, num_heads)
