"""Invoke ``llama-cli`` for GGUF files (no Ollama required)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from stratumllm.exceptions import InferenceError
from stratumllm.resolve import ResolvedModel
from stratumllm.run import InferenceResult, strategy_to_llama_cpp_cli_args
from stratumllm.strategies import DeploymentStrategy


def find_llama_cli(binary: str = "llama-cli") -> str:
    path = shutil.which(binary)
    if path is None:
        raise InferenceError(
            f"`{binary}` not found on PATH. Install llama.cpp or pass `llama_cli=` explicitly."
        )
    return path


def llama_cpp_generate(
    resolved: ResolvedModel,
    prompt: str,
    strategy: DeploymentStrategy,
    *,
    llama_cli: str = "llama-cli",
    max_tokens: int = 256,
    extra_args: list[str] | None = None,
) -> InferenceResult:
    """
    Run ``llama-cli`` against ``resolved.artifact_path``.

    Requires a resolved GGUF path (Ollama blobs work if passed via ``ResolvedModel``).
    """
    if resolved.artifact_path is None:
        raise InferenceError("ResolvedModel has no artifact_path for llama-cli.")

    model_path = Path(resolved.artifact_path)
    if not model_path.is_file():
        raise InferenceError(f"Model file not found: {model_path}")

    cli = find_llama_cli(llama_cli)
    cmd = [
        cli,
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(max_tokens),
        *strategy_to_llama_cpp_cli_args(strategy),
        *(extra_args or []),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise InferenceError("llama-cli timed out.") from exc
    except OSError as exc:
        raise InferenceError(f"Failed to spawn llama-cli: {exc}") from exc

    if result.returncode != 0:
        raise InferenceError(
            f"llama-cli exited {result.returncode}: {result.stderr.strip() or result.stdout}"
        )

    text = result.stdout.strip()
    return InferenceResult(
        text=text,
        model_ref=resolved.name,
        runtime="llama.cpp",
        strategy_kind=strategy.kind.value,
        raw={"stderr": result.stderr, "command": cmd},
    )
