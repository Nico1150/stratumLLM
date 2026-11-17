"""Ollama show quantization parser."""

from __future__ import annotations

from stratumllm.model_spec import QuantizationKind
from stratumllm.resolve.ollama import (
    parse_show_quantization_label,
    quantization_from_show,
)


def test_parse_show_quantization_label() -> None:
    text = """
    Model
      parameters      7.6B
      quantization    Q4_K_M
    """
    assert parse_show_quantization_label(text) == "Q4_K_M"


def test_quantization_from_show() -> None:
    text = "  quantization    q8_0\n"
    assert quantization_from_show(text) is QuantizationKind.Q8_0


def test_quantization_unknown_label() -> None:
    assert quantization_from_show("  quantization    IQ4_XS\n") is None