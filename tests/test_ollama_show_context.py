"""Ollama show context length parser."""

from __future__ import annotations

from stratumllm.resolve.ollama import parse_show_context_length


def test_parse_show_context_length() -> None:
    text = """
    Model
      architecture    llama
      parameters      8.0B
      context length  128000
    """
    assert parse_show_context_length(text) == 128000


def test_parse_missing() -> None:
    assert parse_show_context_length("architecture llama") is None