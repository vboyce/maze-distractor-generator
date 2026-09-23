import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import distract


def run_cli(argv):
    with patch("distract.run_stuff") as run_stuff:
        distract.main(argv)
    return run_stuff.call_args


def test_json_format_is_passed_through():
    call = run_cli(["in.csv", "out.js", "--format", "json", "--module-name", "items"])
    assert call.kwargs["outformat"] == "json"
    assert call.kwargs["module_name"] == "items"


def test_default_format_is_delim():
    assert run_cli(["in.csv", "out.csv"]).kwargs["outformat"] == "delim"


def test_logging_summary_and_model_options_are_passed_through():
    call = run_cli(["in.csv", "out.csv", "--log", "log.csv", "--summary", "summary.csv",
                    "--model", "distilgpt2", "--backend", "transformers_causal", "-p", "params.txt"])
    assert call.kwargs["logfile"] == "log.csv"
    assert call.kwargs["summaryfile"] == "summary.csv"
    assert call.kwargs["model_override"] == "distilgpt2"
    assert call.kwargs["backend_override"] == "transformers_causal"
    assert call.kwargs["parameters"] == "params.txt"


def test_review_options_are_passed_through():
    call = run_cli(["in.csv", "--rejection-file", "review.csv", "--longform", "new.csv", "--num-options", "3"])
    assert call.kwargs["rejection_file"] == "review.csv"
    assert call.kwargs["longform_outfile"] == "new.csv"
    assert call.kwargs["num_options"] == 3
