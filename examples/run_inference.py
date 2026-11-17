#!/usr/bin/env python3
"""Plan deployment and run a single prompt (Ollama or llama-cli)."""

from __future__ import annotations

import sys

from stratumllm.cli import run_main

if __name__ == "__main__":
    sys.exit(run_main())