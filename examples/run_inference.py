#!/usr/bin/env python3
"""Plan deployment and run a single prompt (Ollama or llama-cli)."""

from __future__ import annotations

import argparse
import sys

from stratumllm import run_inference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Ollama model name or path to .gguf")
    parser.add_argument("prompt", nargs="?", default="Why is the sky blue? Answer in one sentence.")
    parser.add_argument(
        "--runtime",
        choices=("auto", "ollama", "llama.cpp"),
        default="auto",
        help="auto: .gguf → llama-cli, else Ollama",
    )
    parser.add_argument("--no-pull", action="store_true", help="Do not run ollama pull if missing")
    args = parser.parse_args(argv)

    resolved, strategy, result = run_inference(
        args.model,
        args.prompt,
        runtime=args.runtime,  # type: ignore[arg-type]
        pull_if_missing=not args.no_pull,
    )
    print(f"--- {result.runtime} ({strategy.kind.value}) ---")
    print(f"Model: {resolved.name} | weights: {resolved.weight_size_gib:.2f} GiB")
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
