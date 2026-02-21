"""Ollama backend for surprisal computation (stub - requires logprobs support)"""


class OllamaBackend:
    """Surprisal backend using Ollama API. Requires Ollama running locally.
    Stub: get_surprisal not yet implemented (needs per-token logprobs from Ollama)."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        raise NotImplementedError(
            "Ollama backend not yet implemented. "
            "Surprisal requires per-token log-probabilities. "
            "Check if your Ollama version exposes logprobs in the completion API."
        )
