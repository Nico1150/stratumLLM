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
    HardwareProfile,
    ModelSpec,
    discover_hardware_profile,
)

hw = discover_hardware_profile()
model = ModelSpec(name="llama3.1", parameter_count_b=8.0, context_length=8192)
strategy = DeploymentStrategyEvaluator().evaluate(hw, model)
print(strategy.kind, strategy.to_llama_cpp_kwargs())
```

## Resolver (Ollama name o GGUF)

```python
from stratumllm import plan_deployment_from_model, resolve_model

resolved, strategy = plan_deployment_from_model("qwen2.5:7b")
# o: plan_deployment_from_model("/path/to/model.gguf")
# Verificación: resolve_model("qwen2.5:7b", expect_parameter_count_b=7.6, strict=True)
```

## Ejecutar inferencia (plan + run)

```python
from stratumllm import run_inference

resolved, strategy, result = run_inference("qwen2.5:7b", "Hello!")
print(result.text)
```

```bash
python examples/run_inference.py qwen2.5:7b "Why is the sky blue?"
```
