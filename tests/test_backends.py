import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backends import load_surprisal_model


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        load_surprisal_model("gpt2", backend="nonsense")


def test_ollama_is_not_offered_until_implemented():
    with pytest.raises(ValueError, match="Unknown backend"):
        load_surprisal_model("llama3", backend="ollama")


def test_a_model_with_no_language_model_head_is_an_error(tmp_path):
    """Loaded as a causal LM, it would get a new, random head."""
    from transformers import GPT2Config
    GPT2Config(architectures=["GPT2Model"]).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="transformers_causal"):
        load_surprisal_model(str(tmp_path), backend="transformers")


def litellm_backend_answering(monkeypatch, token_logprobs):
    """A LiteLLMBackend whose API returns "The cat sat" as three tokens with
    the given log-probabilities (natural log), as text_completion does with
    echo=True."""
    import math
    import types
    fake = types.ModuleType("litellm")

    def text_completion(**kwargs):
        assert kwargs["prompt"] == "The cat sat" and kwargs["echo"] is True
        logprobs = {"tokens": ["The", " cat", " sat"], "text_offset": [0, 3, 7], "token_logprobs": token_logprobs}
        return types.SimpleNamespace(choices=[types.SimpleNamespace(logprobs=logprobs)])

    fake.text_completion = text_completion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return load_surprisal_model("some/model", backend="litellm", prepend_start_token=False)


def test_litellm_surprisal_is_the_word_s_tokens_in_bits(monkeypatch):
    import math
    backend = litellm_backend_answering(monkeypatch, [None, -2.0, -3.0])
    assert backend.get_surprisal("The cat", "sat") == pytest.approx(3.0 / math.log(2))


def test_litellm_missing_log_probability_for_a_word_token_is_an_error(monkeypatch):
    backend = litellm_backend_answering(monkeypatch, [None, -2.0, None])
    with pytest.raises(ValueError, match="no log-probability"):
        backend.get_surprisal("The cat", "sat")
