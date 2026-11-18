# StratumLLM

Gestor de estrategias de despliegue para LLM locales: elige carga completa en GPU, distribución multi-GPU o offloading estratificado. En Apple Silicon (memoria unificada), un solo presupuesto de pool; full Metal si el modelo cabe (`uma_metal_residency_fraction=1.0` por defecto).

## Instalación

```bash
pip install -e ".[nvidia]"
```

## Uso rápido

```python
from stratumllm import (
    DeploymentStrategyEvaluator,
    ModelSpec,
    discover_hardware_profile,
)

hw = discover_hardware_profile()
model = ModelSpec(name="llama3.1", parameter_count_b=8.0, context_length=8192)
strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
print(strategy.kind, strategy.to_llama_cpp_kwargs())
```

El plan resta **KV cache** del presupuesto GPU/unified usando `ModelSpec.context_length` y `kv_cache_storage` (más `kv_cache_reserve_gib` en `HardwareProfile`).

## CLI

```bash
pip install -e .
stratumllm-plan qwen2.5:7b
stratumllm-plan qwen2.5:7b --pull
stratumllm-plan qwen2.5:7b --context 8192 --kv-storage q8_0
stratumllm-plan qwen2.5:7b --json
stratumllm-plan qwen2.5:7b --context-sweep 4096,8192,16384,32768
stratumllm-plan qwen2.5:7b --check-ollama-kv

stratumllm-run qwen2.5:7b "Why is the sky blue?"
stratumllm-run qwen2.5:7b "hola" --json
stratumllm-run model.gguf "hola" --runtime llama.cpp --kv-storage q4_0 --context 8192
```

| Flag | Efecto |
|------|--------|
| `--context` | Longitud para sizing de KV y (en run) `num_ctx` |
| `--kv-storage` | Dtype del KV en el **plan** (`f16`, `q8_0`, `q4_0`) |
| `--context-sweep` | Tabla estrategia vs varios contextos (`4096,8192` o `4096:32768:4096`) |
| `--json` | JSON con hardware, spec, estrategia y sweep (`stratumllm-run --json` incluye la respuesta) |
| `--check-ollama-kv` | Avisa si el proceso local `ollama` no coincide con el plan |
| `--strict-ollama-kv` | Falla si no se puede verificar o hay mismatch |
| `--check-ollama-ps` | Tras `stratumllm-run`, compara `/api/ps` (VRAM y context) |
| `--strict-ollama-ps` | Falla si `/api/ps` no cuadra con el plan |
| `--max-cpu-offload-gib` | Tope de peso en CPU en offload (plan y run) |
| `--uma-metal-fraction` | Fracción del pool unified en Metal |

Ejemplo de sweep:

```bash
stratumllm-plan qwen2.5:7b --context-sweep 4096:131072:8192 --kv-storage q8_0
```

También: `examples/context_sweep.py`.

## Plan vs ejecución (KV cache)

| Runtime | El plan usa `--kv-storage` | La inferencia real |
|---------|---------------------------|---------------------|
| **llama.cpp** | Sí | Sí: `--cache-type-k/v` en `llama-cli` |
| **Ollama** | Sí | **No por request**: global en el daemon vía `OLLAMA_KV_CACHE_TYPE` al `ollama serve` |

StratumLLM imprime los `export` necesarios en plan/run (Ollama). Para alinear memoria real con el plan en Ollama:

```bash
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_FLASH_ATTENTION=1
ollama serve
```

Prioridad del KV en el plan: `--kv-storage` → `OLLAMA_KV_CACHE_TYPE` en el shell de StratumLLM (runtime Ollama) → `f16`.

Contexto del modelo: GGUF / `ollama show` → default `4096` → override con `--context`.

## Resolver (Ollama name o GGUF)

```python
from stratumllm import plan_deployment_from_model, resolve_model

resolved, strategy = plan_deployment_from_model(
    "qwen2.5:7b",
    context_length=8192,
    kv_cache_storage=None,  # o KvCacheStorage.Q8_0
)
# o: plan_deployment_from_model("/path/to/model.gguf", runtime="llama.cpp")
```

## Ejecutar inferencia (plan + run)

```python
from stratumllm import run_inference

resolved, strategy, result = run_inference(
    "qwen2.5:7b",
    "Hello!",
    context_length=8192,
    check_ollama_kv=True,
)
print(result.text)
```

Integradores: `KvCacheStorage`, `resolve_kv_cache_storage`, `build_plan_document`, `build_run_document`, `check_ollama_kv_alignment`, y `parse_context_sweep` se exportan desde `import stratumllm`.

CI corre tests en **Ubuntu** y **macOS** (Python 3.11–3.12).