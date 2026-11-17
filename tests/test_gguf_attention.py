"""GGUF attention metadata → KV planning fields."""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import pytest

from conftest import _write_kv
from stratumllm import resolve_gguf
from stratumllm.resolve.attention import attention_footprint_from_gguf


def _build_llama_kv_gguf(tmp_path: Path) -> Path:
    bio = BytesIO()
    bio.write(b"GGUF")
    bio.write(struct.pack("<I", 3))
    bio.write(struct.pack("<Q", 0))
    bio.write(struct.pack("<Q", 7))
    _write_kv(bio, "general.name", 8, "llama-8b")
    _write_kv(bio, "general.architecture", 8, "llama")
    _write_kv(bio, "llama.block_count", 4, 32)
    _write_kv(bio, "llama.embedding_length", 4, 4096)
    _write_kv(bio, "llama.attention.head_count", 4, 32)
    _write_kv(bio, "llama.attention.head_count_kv", 4, 4)
    _write_kv(bio, "llama.attention.key_length", 4, 128)
    path = tmp_path / "kv.gguf"
    path.write_bytes(bio.getvalue())
    return path


def test_attention_footprint_from_metadata() -> None:
    meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
        "llama.attention.key_length": 128,
    }
    fp = attention_footprint_from_gguf(meta)
    assert fp.hidden_size == 4096
    assert fp.num_kv_heads == 8
    assert fp.head_dim == 128


def test_resolve_populates_model_spec_kv_fields(tmp_path: Path) -> None:
    path = _build_llama_kv_gguf(tmp_path)
    resolved = resolve_gguf(path)
    assert resolved.hidden_size == 4096
    assert resolved.num_kv_heads == 4
    assert resolved.head_dim == 128

    spec = resolved.to_model_spec(context_length=4096)
    heuristic_only = spec.model_copy(
        update={"hidden_size": None, "num_kv_heads": None, "head_dim": None}
    )
    assert spec.estimated_kv_cache_gib() < heuristic_only.estimated_kv_cache_gib()


def test_head_dim_from_embedding_over_head_count() -> None:
    meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 3072,
        "llama.attention.head_count": 24,
    }
    fp = attention_footprint_from_gguf(meta)
    assert fp.head_dim == 128
    assert fp.num_kv_heads == 24


def test_mha_without_gqa_key_uses_head_count() -> None:
    meta = {
        "general.architecture": "llama",
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
    }
    fp = attention_footprint_from_gguf(meta)
    assert fp.num_kv_heads == 32


def test_context_length_from_gguf(tmp_path: Path) -> None:
    bio = BytesIO()
    bio.write(b"GGUF")
    bio.write(struct.pack("<I", 3))
    bio.write(struct.pack("<Q", 0))
    bio.write(struct.pack("<Q", 3))
    _write_kv(bio, "general.name", 8, "m")
    _write_kv(bio, "general.architecture", 8, "llama")
    _write_kv(bio, "llama.context_length", 4, 131072)
    path = tmp_path / "ctx.gguf"
    path.write_bytes(bio.getvalue())
    resolved = resolve_gguf(path)
    assert resolved.context_length == 131072
    assert resolved.to_model_spec().context_length == 131072
    assert resolved.to_model_spec(context_length=8192).context_length == 8192