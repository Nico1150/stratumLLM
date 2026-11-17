#!/usr/bin/env python3
"""Plan deployment from an Ollama model name, a GGUF path, or a manual ModelSpec."""

from __future__ import annotations

import sys

from stratumllm.cli import plan_main

if __name__ == "__main__":
    sys.exit(plan_main())