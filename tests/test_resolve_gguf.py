"""GGUF model resolver tests."""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import pytest

from stratumllm import resolve_gguf
from stratumllm.exceptions import ModelResolutionError
from stratumllm.model_spec import QuantizationKind
from conftest import _write_kv, build_minimal_gguf


def test_resolve_gguf_uses_file_size_and_metadata(tmp_path: Path) -> None:
    path = build_minimal_gguf(tmp_path, extra_payload=b"\x00" * (4 * 1024**2))
    resolved = resolve_gguf(path)

    assert resolved.name == "llama-3.1-8b-instruct"
    assert resolved.num_layers == 32
    assert resolved.architecture == "llama"
    assert resolved.parameter_count_b == pytest.approx(8.0)
    assert resolved.weight_size_gib == pytest.approx((path.stat().st_size) / (1024**3))

    spec = resolved.to_model_spec()
    assert spec.weight_size_gib == resolved.weight_size_gib
    assert spec.estimated_weight_gib() == resolved.weight_size_gib
    assert spec.resolved_num_layers() == 32


def test_resolve_gguf_file_type_quantization(tmp_path: Path) -> None:
    bio = BytesIO()
    bio.write(b"GGUF")
    bio.write(struct.pack("<I", 3))
    bio.write(struct.pack("<Q", 0))
    bio.write(struct.pack("<Q", 2))
    _write_kv(bio, "general.name", 8, "m")
    _write_kv(bio, "general.file_type", 4, 15)
    path = tmp_path / "q4.gguf"
    path.write_bytes(bio.getvalue())

    resolved = resolve_gguf(path)
    assert resolved.quantization == QuantizationKind.Q4_K_M


def test_resolve_missing_file() -> None:
    with pytest.raises(ModelResolutionError):
        resolve_gguf("/no/such/model.gguf")
