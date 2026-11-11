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
