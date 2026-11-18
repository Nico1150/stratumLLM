"""CLI hooks for syncing Ollama KV env with ``--kv-storage``."""

from __future__ import annotations

import argparse
import sys

from stratumllm.kv_cache import KvCacheStorage
from stratumllm.run.ollama_env import sync_ollama_kv_from_plan, write_ollama_env_file
from stratumllm.runtime import RuntimeKind


def effective_check_ollama_kv(args: argparse.Namespace, runtime: RuntimeKind) -> bool:
    """Con ``--kv-storage`` en Ollama, verificar el daemon salvo ``--no-check-ollama-kv``."""
    if runtime != "ollama":
        return False
    if getattr(args, "no_check_ollama_kv", False):
        return False
    if args.check_ollama_kv:
        return True
    return bool(args.kv_storage)


def sync_ollama_kv_if_explicit(
    *,
    kv_flag: str | None,
    storage: KvCacheStorage,
    runtime: RuntimeKind,
    restart_ollama: bool,
    as_json: bool = False,
) -> None:
    """When the user passed ``--kv-storage``, apply Ollama global env to match the plan."""
    if runtime != "ollama" or not kv_flag:
        return
    result = sync_ollama_kv_from_plan(storage, restart_if_running=restart_ollama)
    if as_json or not result.notes:
        return
    for line in result.notes:
        print(f"Ollama KV: {line}", file=sys.stderr)


def write_ollama_env_if_requested(
    *,
    path: str | None,
    storage: KvCacheStorage,
    runtime: RuntimeKind,
    as_json: bool,
) -> None:
    if not path or runtime != "ollama":
        return
    written = write_ollama_env_file(path, storage)
    if not as_json:
        print(f"Ollama KV env escrito en {written} (source antes de ollama serve)", file=sys.stderr)