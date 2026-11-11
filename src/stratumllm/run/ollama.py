"""Ollama HTTP API client for generation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from stratumllm.exceptions import InferenceError, ModelResolutionError
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
) -> InferenceResult:
    """
    Run ``/api/generate`` with strategy-driven ``num_gpu`` (and related) options.

    Does not mutate the Ollama daemon environment; placement is request-scoped via ``options``.
    """
    if stream:
        raise InferenceError("Streaming is not implemented; use stream=False.")

    base = (base_url or ollama_base_url()).rstrip("/")
    merged_options = {**strategy_to_ollama_options(strategy), **(options or {})}
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
