# Changelog

## 0.2.0 — 2026-03-24

### Added

- KV cache planning: `--kv-storage` (`f16`, `q8_0`, `q4_0`), `OLLAMA_KV_CACHE_TYPE` resolution, llama.cpp cache flags.
- Ollama KV sync: `sync_ollama_kv_from_plan`, `--restart-ollama`, `--write-ollama-env`, auto `--check-ollama-kv` with `--kv-storage`.
- CLI: `--json` (plan/run), `--context-sweep`, evaluator policy flags, `/api/ps` checks.
- CI on Ubuntu, macOS, and Windows; public API exports; Spanish mismatch guidance.

### Changed

- `resolve_ollama` fills quantization from `ollama show` when GGUF lacks it.

## 0.1.0

Initial release: hardware discovery, deployment strategies, Ollama/GGUF resolve, plan/run CLIs.