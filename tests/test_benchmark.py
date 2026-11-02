import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import benchmark


def test_a_model_that_fails_stops_the_benchmark(monkeypatch, tmp_path):
    """A failed run used to be logged and recorded as a status string, and the
    benchmark went on, so a broken model looked like a finished benchmark."""
    monkeypatch.chdir(tmp_path)
    calls = []

    def failing_run_stuff(**kwargs):
        calls.append(kwargs["model_override"])
        raise RuntimeError("model would not load")

    monkeypatch.setattr(benchmark, "run_stuff", failing_run_stuff)
    with pytest.raises(RuntimeError, match="model would not load"):
        benchmark.run_benchmark(models=[("first", "transformers"), ("second", "transformers")],
                                results_file=str(tmp_path / "results.csv"))
    assert calls == ["first"]
