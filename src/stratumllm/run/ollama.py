"""Ollama HTTP API client for generation."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import psutil

from stratumllm.exceptions import ConfigurationError, InferenceError, ModelResolutionError
from stratumllm.kv_cache import (
    KvCacheStorage,
    format_ollama_kv_mismatch_lines,
    ollama_serve_command,
    parse_kv_cache_storage,
)
from stratumllm.resolve.ollama import ensure_model_pulled as _ensure_model_pulled
from stratumllm.run import InferenceResult, strategy_to_ollama_options
from stratumllm.strategies import DeploymentStrategy

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


def ollama_base_url() -> str:
    import os

    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL).rstrip("/")


def ensure_ollama_available(base_url: str | None = None) -> None:
    """Raise if the Ollama API does not respond."""
    base = (base_url or ollama_base_url()).rstrip("/")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise InferenceError(
            f"Ollama is not reachable at {base}. Start the daemon (`ollama serve`) or set OLLAMA_HOST."
        ) from exc


@dataclass(frozen=True, slots=True)
class OllamaKvProbeResult:
    """Best-effort read of ``OLLAMA_KV_CACHE_TYPE`` from a local ``ollama`` process."""

    status: Literal["probed", "no_process", "unreadable"]
    server_storage: KvCacheStorage | None = None


def _kv_from_psutil() -> KvCacheStorage | Literal["no_process", "unreadable"]:
    """Cross-platform env scan (Linux, macOS, Windows) via psutil."""
    saw_ollama = False
    unreadable = False
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or proc.name() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if "ollama" not in name:
            continue
        saw_ollama = True
        try:
            raw = proc.environ().get("OLLAMA_KV_CACHE_TYPE", "").strip()
        except (psutil.AccessDenied, psutil.ZombieProcess):
            unreadable = True
            continue
        if raw:
            return parse_kv_cache_storage(raw)
        return KvCacheStorage.F16
    if not saw_ollama:
        return "no_process"
    if unreadable:
        return "unreadable"
    return "unreadable"


def _find_ollama_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-x", "ollama"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _kv_from_linux_environ(pid: int) -> KvCacheStorage | Literal["unreadable"]:
    path = Path(f"/proc/{pid}/environ")
    try:
        raw = path.read_bytes()
    except OSError:
        return "unreadable"
    for entry in raw.split(b"\0"):
        if entry.startswith(b"OLLAMA_KV_CACHE_TYPE="):
            value = entry.split(b"=", 1)[1].decode(errors="replace")
            return parse_kv_cache_storage(value)
    return KvCacheStorage.F16


def _kv_from_darwin_ps(pid: int) -> KvCacheStorage | Literal["unreadable"]:
    try:
        result = subprocess.run(
            ["ps", "eww", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unreadable"
    if result.returncode != 0 or str(pid) not in result.stdout:
        return "unreadable"
    match = re.search(r"OLLAMA_KV_CACHE_TYPE=(\S+)", result.stdout)
    if match:
        return parse_kv_cache_storage(match.group(1))
    return KvCacheStorage.F16


def probe_ollama_server_kv_cache_type() -> OllamaKvProbeResult:
    """
    Inspect the local ``ollama`` process environment (not exposed over HTTP).

    When the variable is unset on the server, Ollama defaults to ``f16``.
    """
    psutil_outcome = _kv_from_psutil()
    if psutil_outcome not in ("no_process", "unreadable"):
        return OllamaKvProbeResult(status="probed", server_storage=psutil_outcome)
    if psutil_outcome == "no_process":
        return OllamaKvProbeResult(status="no_process")

    pids = _find_ollama_pids()
    if not pids:
        return OllamaKvProbeResult(status="no_process")
    system = platform.system()
    for pid in pids:
        if system == "Linux":
            outcome = _kv_from_linux_environ(pid)
        elif system == "Darwin":
            outcome = _kv_from_darwin_ps(pid)
        elif system == "Windows":
            continue
        else:
            return OllamaKvProbeResult(status="unreadable")
        if outcome == "unreadable":
            continue
        return OllamaKvProbeResult(status="probed", server_storage=outcome)
    return OllamaKvProbeResult(status="unreadable")


def check_ollama_kv_alignment(
    planned: KvCacheStorage,
    *,
    strict: bool = False,
) -> list[str]:
    """Return warning lines when the running server KV dtype may not match the plan.

    With ``strict=True``, raise ``ConfigurationError`` on mismatch or if unprobed.
    """
    probe = probe_ollama_server_kv_cache_type()
    if probe.status == "no_process":
        msg = (
            "Could not find a local `ollama` process to read OLLAMA_KV_CACHE_TYPE "
            "(the HTTP API does not report it). Ensure the server was started with "
            f"`{ollama_serve_command(planned)}` if you rely on this plan's KV size."
        )
        if strict:
            raise ConfigurationError(msg)
        return [f"warning: {msg}"]
    if probe.status == "unreadable" or probe.server_storage is None:
        msg = (
            "Could not read OLLAMA_KV_CACHE_TYPE from the ollama process "
            "(permissions or unsupported host). Verify the server matches the plan "
            f"({planned.value}) manually."
        )
        if strict:
            raise ConfigurationError(msg)
        return [f"warning: {msg}"]
    if probe.server_storage != planned:
        lines = format_ollama_kv_mismatch_lines(planned, probe.server_storage)
        if strict:
            raise ConfigurationError(lines[0].removeprefix("warning: "))
        return lines
    return []


def ensure_model_pulled(model_ref: str, *, pull_if_missing: bool = True) -> None:
    """See ``stratumllm.resolve.ollama.ensure_model_pulled`` (raises ``InferenceError``)."""
    try:
        _ensure_model_pulled(model_ref, pull_if_missing=pull_if_missing)
    except ModelResolutionError as exc:
        raise InferenceError(str(exc)) from exc


def ollama_generate(
    model_ref: str,
    prompt: str,
    strategy: DeploymentStrategy,
    *,
    base_url: str | None = None,
    options: dict[str, Any] | None = None,
    stream: bool = False,
    num_ctx: int | None = None,
) -> InferenceResult:
    """
    Run ``/api/generate`` with strategy-driven ``num_gpu`` (and related) options.

    ``num_ctx`` should match the context length used during deployment planning.
    """
    if stream:
        raise InferenceError("Streaming is not implemented; use stream=False.")

    base = (base_url or ollama_base_url()).rstrip("/")
    merged_options = {**strategy_to_ollama_options(strategy), **(options or {})}
    if num_ctx is not None:
        merged_options["num_ctx"] = num_ctx
    payload = {
        "model": model_ref,
        "prompt": prompt,
        "stream": False,
        "options": merged_options,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise InferenceError(f"Ollama generate failed ({exc.code}): {detail}") from exc
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        raise InferenceError(f"Ollama generate request failed: {exc}") from exc

    text = body.get("response", "")
    if not isinstance(text, str):
        raise InferenceError("Ollama response missing 'response' text field.")

    return InferenceResult(
        text=text,
        model_ref=model_ref,
        runtime="ollama",
        strategy_kind=strategy.kind.value,
        raw=body,
    )
