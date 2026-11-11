"""Ollama name → blob resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stratumllm.exceptions import ModelResolutionError
from stratumllm.resolve.ollama import parse_ollama_model_ref, resolve_ollama_blob_path
from stratumllm.resolve import resolve_ollama
from conftest import build_minimal_gguf


def test_parse_model_ref() -> None:
    assert parse_ollama_model_ref("qwen2.5:7b") == ("qwen2.5", "7b")
    assert parse_ollama_model_ref("llama3.2") == ("llama3.2", "latest")


def test_resolve_ollama_layout(tmp_path: Path) -> None:
    gguf = build_minimal_gguf(tmp_path)
    digest_hex = "deadbeef" * 8
    blob = tmp_path / "blobs" / f"sha256-{digest_hex}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(gguf.read_bytes())

    manifest_dir = (
        tmp_path / "manifests" / "registry.ollama.ai" / "library" / "demo" / "7b"
    )
    manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": f"sha256:{digest_hex}",
                "size": blob.stat().st_size,
            }
        ]
    }
    manifest_dir.write_text(json.dumps(manifest), encoding="utf-8")

    path, canonical = resolve_ollama_blob_path("demo:7b", models_root=tmp_path)
    assert path == blob
    assert canonical == "demo:7b"

    resolved = resolve_ollama("demo:7b", models_root=tmp_path)
    assert resolved.weight_size_gib > 0


def test_strict_parameter_mismatch(tmp_path: Path) -> None:
    gguf = build_minimal_gguf(tmp_path)
    digest_hex = "cafebabe" * 8
    blob = tmp_path / "blobs" / f"sha256-{digest_hex}"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(gguf.read_bytes())
    manifest_dir = (
        tmp_path / "manifests" / "registry.ollama.ai" / "library" / "demo" / "7b"
    )
    manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": f"sha256:{digest_hex}",
                        "size": blob.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelResolutionError):
        resolve_ollama(
            "demo:7b",
            expect_parameter_count_b=70.0,
            strict=True,
            models_root=tmp_path,
        )
