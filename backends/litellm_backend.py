"""LiteLLM backend for surprisal computation.

Uses litellm.text_completion with echo=True to get per-token logprobs,
allowing surprisal computation across any provider liteLLM supports
(OpenAI, Azure, vLLM, Ollama via OpenAI-compat, etc.).
"""
import math
import numpy as np
from typing import Optional


class LiteLLMBackend:
    """Surprisal backend using liteLLM's text_completion API.

    Requires `pip install litellm` and appropriate API keys/config
    for the chosen provider.

    Args:
        model_name: liteLLM model string (e.g. 'text-davinci-003',
            'openai/gpt-3.5-turbo-instruct', 'ollama/llama2')
        api_base: Optional API base URL for self-hosted providers
        prepend_start_token: Must be False: the API is sent text, so no start
            token can be put before it. (True is the default for every
            backend, so that the error says to change it.)
        revision: Must be None: model versions are set by the provider.
    """

    def __init__(self, model_name: str, api_base: Optional[str] = None,
                 prepend_start_token: bool = True, revision: Optional[str] = None):
        if prepend_start_token:
            raise ValueError("The litellm backend cannot put a start token before the text it sends; "
                             "set prepend_start_token: False")
        if revision is not None:
            raise ValueError("The litellm backend cannot pin a model revision; leave revision unset")
        self.model_commit = None  # for the run record: the provider sets the model version
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "litellm package is required for LiteLLMBackend. "
                "Install with: pip install litellm"
            )
        self.model_name = model_name
        self.api_base = api_base

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        from litellm import text_completion

        prefix = prefix.strip()
        word = word.strip()
        if not prefix:
            raise ValueError("prefix must not be empty; surprisal requires at least one token of context")
        full_text = prefix + " " + word

        kwargs = dict(
            model=self.model_name,
            prompt=full_text,
            echo=True,
            max_tokens=0,
            logprobs=1,
        )
        if self.api_base:
            kwargs["api_base"] = self.api_base

        response = text_completion(**kwargs)
        logprobs_data = response.choices[0].logprobs

        if logprobs_data is None:
            raise RuntimeError(
                f"Model {self.model_name} did not return logprobs. "
                "Ensure the provider supports the logprobs parameter."
            )

        tokens = logprobs_data["tokens"]
        token_logprobs = logprobs_data["token_logprobs"]
        text_offset = logprobs_data["text_offset"]

        word_start = len(prefix)
        word_token_indices = [i for i, off in enumerate(text_offset) if off >= word_start]
        if not word_token_indices:
            raise ValueError(
                f"Could not find word '{word}' tokens in logprobs response. "
                f"Tokens: {tokens}, offsets: {text_offset}"
            )

        log_prob_sum = 0.0
        for i in word_token_indices:
            lp = token_logprobs[i]
            if lp is None:
                raise ValueError(f"The API returned no log-probability for token {tokens[i]!r} of {word!r}")
            if not np.isfinite(lp):
                raise ValueError(
                    f"Model assigned zero probability to token {repr(tokens[i])} "
                    f"at position {i}"
                )
            log_prob_sum += lp

        return -log_prob_sum / math.log(base)
