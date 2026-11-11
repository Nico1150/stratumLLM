"""Shared test helpers."""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path


def _pack_string(bio: BytesIO, text: str) -> None:
    data = text.encode("utf-8")
    bio.write(struct.pack("<Q", len(data)))
    bio.write(data)


def _write_kv(bio: BytesIO, key: str, value_type: int, value: object) -> None:
    _pack_string(bio, key)
    bio.write(struct.pack("<I", value_type))
    if value_type == 8:
        assert isinstance(value, str)
        _pack_string(bio, value)
    elif value_type == 4:
        bio.write(struct.pack("<I", int(value)))
    else:
        raise ValueError(value_type)


def build_minimal_gguf(tmp_path: Path, *, extra_payload: bytes = b"") -> Path:
    bio = BytesIO()
    bio.write(b"GGUF")
    bio.write(struct.pack("<I", 3))
    bio.write(struct.pack("<Q", 0))
    bio.write(struct.pack("<Q", 3))
    _write_kv(bio, "general.name", 8, "llama-3.1-8b-instruct")
    _write_kv(bio, "general.architecture", 8, "llama")
    _write_kv(bio, "llama.block_count", 4, 32)
    bio.write(extra_payload)
    path = tmp_path / "test.gguf"
    path.write_bytes(bio.getvalue())
    return path
