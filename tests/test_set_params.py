import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from set_params import set_params, DEFAULTS


def test_no_source_gives_all_defaults():
    assert set_params(None) == DEFAULTS


def test_dict_values_override_defaults():
    params = set_params({"min_delta": 5, "model": "distilgpt2"})
    assert params["min_delta"] == 5
    assert params["model"] == "distilgpt2"
    assert params["min_abs"] == DEFAULTS["min_abs"]


def test_reads_params_file_and_skips_comments(tmp_path):
    f = tmp_path / "params.txt"
    f.write_text('# a comment\nmin_delta: 12\nmodel: "EleutherAI/pythia-160m"\n')
    params = set_params(str(f))
    assert params["min_delta"] == 12
    assert params["model"] == "EleutherAI/pythia-160m"


def test_word_list_keys_are_accepted(tmp_path):
    params = set_params({"include_words": "curated_word_list.txt", "exclude_words": "exclude.txt"})
    assert params["include_words"] == "curated_word_list.txt"


def test_unknown_key_raises():
    """Old params files use keys like model_loc that the code no longer reads;
    silently ignoring them would run with the wrong model."""
    with pytest.raises(ValueError, match="model_loc"):
        set_params({"model_loc": "french_data/model.pt"})
