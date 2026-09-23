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
