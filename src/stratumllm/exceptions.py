"""Domain exceptions for StratumLLM."""


class StratumLLMError(Exception):
    """Base error for this package."""


class InsufficientResourcesError(StratumLLMError):
    """Raised when the model cannot be placed under current hardware constraints."""


class HardwareDiscoveryError(StratumLLMError):
    """Raised when automatic hardware probing fails irrecoverably."""


class ConfigurationError(StratumLLMError):
    """Raised when user-supplied configuration is inconsistent."""


class ModelResolutionError(StratumLLMError):
    """Raised when a model artifact cannot be resolved to planning metadata."""


class InferenceError(StratumLLMError):
    """Raised when a local runtime (Ollama, llama.cpp) fails to run inference."""
