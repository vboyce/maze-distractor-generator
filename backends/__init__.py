"""Surprisal model backends (transformers causal/masked, ollama, lightllm)"""
import logging
from backends.base import SurprisalBackend
from backends.transformers_backend import CausalLMBackend, MaskedLMBackend

__all__ = [
    "SurprisalBackend",
    "CausalLMBackend",
    "MaskedLMBackend",
    "load_surprisal_model",
]


def _detect_transformers_type(model_name: str) -> str:
    """Use the model config to determine whether a model is causal or masked."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    architectures = getattr(config, "architectures", []) or []
    for arch in architectures:
        if "CausalLM" in arch or "LMHeadModel" in arch:
            return "causal"
        if "MaskedLM" in arch:
            return "masked"
    logging.warning(
        "Could not auto-detect model type from architectures %s; defaulting to causal",
        architectures,
    )
    return "causal"


def load_surprisal_model(
    model_name: str,
    backend: str = "transformers",
    **kwargs
) -> SurprisalBackend:
    """Load a surprisal backend for the given model.

    Args:
        model_name: Model identifier (e.g. 'gpt2', 'bert-base-uncased')
        backend: One of:
            'transformers'        - auto-detect causal vs masked
            'transformers_causal' - force causal LM (GPT-2, Llama, etc.)
            'transformers_masked' - force masked LM (BERT, RoBERTa, etc.)
            'ollama'              - Ollama inference server (stub)
            'lightllm'            - LightLLM inference server (stub)
        **kwargs: Backend-specific options (e.g. device for transformers, base_url for ollama)

    Returns:
        SurprisalBackend instance with get_surprisal(prefix, word, base=2.0)
    """
    if backend == "transformers":
        model_type = _detect_transformers_type(model_name)
        logging.info("Auto-detected %s as %s LM", model_name, model_type)
        if model_type == "masked":
            return MaskedLMBackend(model_name, **kwargs)
        return CausalLMBackend(model_name, **kwargs)
    elif backend == "transformers_causal":
        return CausalLMBackend(model_name, **kwargs)
    elif backend == "transformers_masked":
        return MaskedLMBackend(model_name, **kwargs)
    elif backend == "ollama":
        from backends.ollama_backend import OllamaBackend
        return OllamaBackend(model_name, **kwargs)
    elif backend == "lightllm":
        from backends.lightllm_backend import LightLLMBackend
        return LightLLMBackend(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
