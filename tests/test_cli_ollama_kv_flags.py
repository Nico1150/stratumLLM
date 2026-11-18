"""Auto check-ollama-kv when --kv-storage set."""

from __future__ import annotations

import argparse

from stratumllm.cli_ollama_kv import effective_check_ollama_kv


def _args(**kwargs: object) -> argparse.Namespace:
    base = {
        "kv_storage": None,
        "check_ollama_kv": False,
        "no_check_ollama_kv": False,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_auto_check_with_kv_storage() -> None:
    assert effective_check_ollama_kv(_args(kv_storage="q8_0"), "ollama") is True


def test_no_check_opt_out() -> None:
    assert (
        effective_check_ollama_kv(_args(kv_storage="q8_0", no_check_ollama_kv=True), "ollama")
        is False
    )


def test_no_check_on_llama_cpp() -> None:
    assert effective_check_ollama_kv(_args(kv_storage="q8_0"), "llama.cpp") is False