"""Minimal GGUF header/metadata reader (stdlib only, no tensor load)."""

from __future__ import annotations

import struct
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFReader:
    """Parse GGUF metadata key-value section from the start of a file."""

    _STRUCT_U32 = struct.Struct("<I")
    _STRUCT_U64 = struct.Struct("<Q")
    _STRUCT_I32 = struct.Struct("<i")
    _STRUCT_F32 = struct.Struct("<f")

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        magic = stream.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        self.version = self._read_u32()
        if self.version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {self.version}")
        self.tensor_count = self._read_u64()
        self.metadata_kv_count = self._read_u64()
        self.metadata: dict[str, Any] = {}
        for _ in range(self.metadata_kv_count):
            key = self._read_string()
            value_type = GGUFValueType(self._read_u32())
            self.metadata[key] = self._read_value(value_type)

    def _read(self, size: int) -> bytes:
        data = self._stream.read(size)
        if len(data) != size:
            raise ValueError("Unexpected EOF while parsing GGUF")
        return data

    def _read_u32(self) -> int:
        return self._STRUCT_U32.unpack(self._read(4))[0]

    def _read_u64(self) -> int:
        return self._STRUCT_U64.unpack(self._read(8))[0]

    def _read_string(self) -> str:
        length = self._read_u64()
        return self._read(length).decode("utf-8")

    def _read_value(self, value_type: GGUFValueType) -> Any:
        if value_type == GGUFValueType.UINT8:
            return self._read(1)[0]
        if value_type == GGUFValueType.INT8:
            return struct.unpack("b", self._read(1))[0]
        if value_type == GGUFValueType.UINT16:
            return struct.unpack("<H", self._read(2))[0]
        if value_type == GGUFValueType.INT16:
            return struct.unpack("<h", self._read(2))[0]
        if value_type == GGUFValueType.UINT32:
            return self._read_u32()
        if value_type == GGUFValueType.INT32:
            return self._STRUCT_I32.unpack(self._read(4))[0]
        if value_type == GGUFValueType.FLOAT32:
            return self._STRUCT_F32.unpack(self._read(4))[0]
        if value_type == GGUFValueType.BOOL:
            return bool(self._read(1)[0])
        if value_type == GGUFValueType.STRING:
            return self._read_string()
        if value_type == GGUFValueType.UINT64:
            return self._read_u64()
        if value_type == GGUFValueType.INT64:
            return struct.unpack("<q", self._read(8))[0]
        if value_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", self._read(8))[0]
        if value_type == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(self._read_u32())
            count = self._read_u64()
            return [self._read_value(elem_type) for _ in range(count)]
        raise ValueError(f"Unhandled GGUF value type: {value_type}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Return metadata KV dict from a GGUF file."""
    with path.open("rb") as fh:
        return GGUFReader(fh).metadata
