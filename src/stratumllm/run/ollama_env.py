"""Apply planned KV cache settings to Ollama's global environment (best effort)."""

from __future__ import annotations

import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import psutil

from stratumllm.exceptions import ConfigurationError
from stratumllm.kv_cache import KvCacheStorage, ollama_serve_command, ollama_serve_env_hints
from stratumllm.run.ollama import ollama_base_url


@dataclass(frozen=True, slots=True)
class OllamaKvEnvSyncResult:
    """Outcome of syncing ``OLLAMA_KV_CACHE_TYPE`` from the plan."""

    applied_to_process: bool
    launchctl_ok: bool
    restart_attempted: bool
    notes: list[str]


def apply_ollama_kv_to_process(storage: KvCacheStorage) -> dict[str, str]:
    """Set ``OLLAMA_*`` on this Python process (child ``ollama`` CLIs inherit it)."""
    hints = ollama_serve_env_hints(storage)
    for key, value in hints.items():
        os.environ[key] = value
    return hints


def apply_ollama_kv_launchctl(storage: KvCacheStorage) -> list[str]:
    """macOS: variables visibles para la app de escritorio en el próximo arranque."""
    if platform.system() != "Darwin":
        return []
    notes: list[str] = []
    for key, value in ollama_serve_env_hints(storage).items():
        try:
            subprocess.run(
                ["launchctl", "setenv", key, value],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            notes.append(f"launchctl setenv {key}={value} falló: {exc}")
    return notes


def _wait_ollama_api(base_url: str, *, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    url = f"{base_url.rstrip('/')}/api/tags"
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    return False


def restart_ollama_serve(storage: KvCacheStorage, *, base_url: str | None = None) -> None:
    """
    Terminate local ``ollama`` processes and start ``ollama serve`` with planned KV env.

    Destructive: unloads models and stops any in-flight generation. Use only when requested.
    """
    hints = apply_ollama_kv_to_process(storage)
    env = {**os.environ, **hints}
    base = (base_url or ollama_base_url()).rstrip("/")

    targets: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            name = (proc.info.get("name") or proc.name() or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if "ollama" not in name:
            continue
        targets.append(proc)
    for proc in targets:
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _gone, alive = psutil.wait_procs(targets, timeout=8)
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    creationflags = 0
    if platform.system() == "Windows":
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=platform.system() != "Windows",
        creationflags=creationflags,
    )
    if not _wait_ollama_api(base):
        raise ConfigurationError(
            f"`ollama serve` no respondió en {base} tras reinicio con KV {storage.value}."
        )


def write_ollama_env_file(path: str | Path, storage: KvCacheStorage) -> Path:
    """
    Write a shell script ``source``-able file plus commented launchctl hints.

    Example: ``source ./ollama-kv.env && ollama serve``
    """
    target = Path(path).expanduser()
    hints = ollama_serve_env_hints(storage)
    lines = [
        "#!/bin/sh",
        "# StratumLLM: Ollama global KV env (source before `ollama serve`)",
        "",
    ]
    for key, value in hints.items():
        lines.append(f"export {key}={value}")
    lines.extend(
        [
            "",
            f"# One-shot: {ollama_serve_command(storage)}",
            "# macOS desktop app (after quit Ollama):",
        ]
    )
    for key, value in hints.items():
        lines.append(f"# launchctl setenv {key} {value}")
    lines.append("# Windows (persistent user env): setx " + " ; setx ".join(
        f"{k} {v}" for k, v in hints.items()
    ))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def sync_ollama_kv_from_plan(
    storage: KvCacheStorage,
    *,
    restart_if_running: bool = False,
    base_url: str | None = None,
) -> OllamaKvEnvSyncResult:
    """
    Align global Ollama KV env with the plan when the user chose ``--kv-storage``.

    - Always: ``os.environ`` in this process.
    - macOS: ``launchctl setenv`` for the desktop app on next launch.
    - Optional: ``restart_ollama_serve`` when ``restart_if_running``.
    """
    apply_ollama_kv_to_process(storage)
    launch_notes = apply_ollama_kv_launchctl(storage)
    notes = list(launch_notes)
    restarted = False
    if restart_if_running:
        restart_ollama_serve(storage, base_url=base_url)
        restarted = True
        notes.append(
            f"Daemon Ollama reiniciado con OLLAMA_KV_CACHE_TYPE={storage.llama_cpp_cache_type()}."
        )
    elif platform.system() == "Darwin":
        notes.append(
            "Si Ollama ya estaba abierto (app o serve), cerralo y volvé a abrirlo "
            "para que tome las variables."
        )
    elif not restart_if_running:
        notes.append(
            "Si ``ollama serve`` ya corría, reinicialo con el mismo OLLAMA_KV_CACHE_TYPE "
            "para alinear VRAM con el plan."
        )
    return OllamaKvEnvSyncResult(
        applied_to_process=True,
        launchctl_ok=not launch_notes,
        restart_attempted=restarted,
        notes=notes,
    )