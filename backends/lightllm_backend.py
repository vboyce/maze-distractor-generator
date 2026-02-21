"""LightLLM backend for surprisal computation (stub - requires logprobs support)"""
from typing import Optional


class LightLLMBackend:
    """Surprisal backend using LightLLM inference server.
    Stub: get_surprisal not yet implemented (needs per-token logprobs from API)."""

    def __init__(self, model_name: str, base_url: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.base_url = base_url

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        raise NotImplementedError(
            "LightLLM backend not yet implemented. "
            "Surprisal requires per-token log-probabilities. "
            "Check LightLLM docs for logprobs in completion responses."
        )
