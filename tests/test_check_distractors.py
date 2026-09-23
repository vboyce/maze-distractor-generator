import sys
import os
import importlib
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_importing_the_checker_makes_no_api_calls(monkeypatch):
    fake_litellm = types.ModuleType("litellm")

    def completion(**kwargs):
        raise AssertionError("an LLM API call was made at import time")

    fake_litellm.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("check_distractors", None)
    module = importlib.import_module("check_distractors")
    assert hasattr(module, "benchmark_grammaticality")
