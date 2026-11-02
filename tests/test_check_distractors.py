import sys
import os
import importlib
import types

import pytest

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


def import_checker_with_fake_litellm(monkeypatch):
    """Import check_distractors with a litellm whose completion() fails the test if called."""
    fake_litellm = types.ModuleType("litellm")

    def completion(**kwargs):
        raise AssertionError("an LLM API call was made")

    fake_litellm.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    sys.modules.pop("check_distractors", None)
    return importlib.import_module("check_distractors")


def test_anthropic_model_without_key_fails_before_any_api_call(monkeypatch, tmp_path):
    checker = import_checker_with_fake_litellm(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence,distractors,labels\n"
                      "a,1,The cat sat.,x-x-x dog ran.,0 1 2\n")

    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        checker.benchmark_grammaticality(str(infile), str(tmp_path / "out.csv"),
                                         model="anthropic/claude-sonnet-4-20250514",
                                         env_file=tmp_path / "missing.env")


def test_missing_key_error_names_the_env_file(monkeypatch, tmp_path):
    checker = import_checker_with_fake_litellm(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"

    with pytest.raises(SystemExit, match=str(env_file)):
        checker.require_api_key("anthropic/claude-sonnet-4-20250514", env_file)


def test_key_is_read_from_the_env_file(monkeypatch, tmp_path):
    checker = import_checker_with_fake_litellm(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=test-key-from-file\n")

    checker.require_api_key("anthropic/claude-sonnet-4-20250514", env_file)

    assert os.environ["ANTHROPIC_API_KEY"] == "test-key-from-file"


def test_non_anthropic_model_does_not_need_the_key(monkeypatch, tmp_path):
    checker = import_checker_with_fake_litellm(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    checker.require_api_key("ollama/llama3", tmp_path / "missing.env")
