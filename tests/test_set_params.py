import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from set_params import set_params, DEFAULTS


def test_keys_left_out_get_the_defaults():
    assert set_params({"model": "gpt2"}) == {**DEFAULTS, "model": "gpt2",
                                             "avoid_names_and_abbreviations": True}


def test_dict_values_override_defaults():
    params = set_params({"min_delta": 5, "model": "distilgpt2"})
    assert params["min_delta"] == 5
    assert params["model"] == "distilgpt2"
    assert params["min_abs"] == DEFAULTS["min_abs"]


def test_model_has_no_default():
    """A run with no model named would silently use whichever one was the default."""
    with pytest.raises(ValueError, match="model"):
        set_params({"min_delta": 5})


def test_no_source_and_no_model_raises():
    with pytest.raises(ValueError, match="model"):
        set_params(None)


def test_overrides_supply_a_missing_model():
    assert set_params(None, overrides={"model": "distilgpt2"})["model"] == "distilgpt2"


def test_overrides_win_over_the_source():
    params = set_params({"model": "gpt2"}, overrides={"model": "distilgpt2"})
    assert params["model"] == "distilgpt2"


def test_override_of_none_is_ignored():
    """CLI flags that weren't given arrive as None."""
    params = set_params({"model": "gpt2"}, overrides={"model": None})
    assert params["model"] == "gpt2"


def test_reads_params_file_and_skips_comments(tmp_path):
    f = tmp_path / "params.txt"
    f.write_text('# a comment\nmin_delta: 12\nmodel: "EleutherAI/pythia-160m"\n')
    params = set_params(str(f))
    assert params["min_delta"] == 12
    assert params["model"] == "EleutherAI/pythia-160m"


def test_word_list_keys_are_accepted(tmp_path):
    params = set_params({"model": "gpt2", "include_words": "curated_word_list.txt", "exclude_words": "exclude.txt"})
    assert params["include_words"] == "curated_word_list.txt"


def test_unknown_key_raises():
    """Old params files use keys like model_loc that the code no longer reads;
    silently ignoring them would run with the wrong model."""
    with pytest.raises(ValueError, match="model_loc"):
        set_params({"model": "gpt2", "model_loc": "french_data/model.pt"})


def test_fallback_ranking_defaults_to_gap():
    assert set_params({"model": "gpt2"})["fallback_ranking"] == "gap"


def test_unknown_fallback_ranking_raises_before_the_run():
    with pytest.raises(ValueError, match="fallback_ranking"):
        set_params({"model": "gpt2", "fallback_ranking": "biggest"})


def test_start_token_is_prepended_by_default():
    assert set_params({"model": "gpt2"})["prepend_start_token"] is True


def test_revision_defaults_to_none():
    assert set_params({"model": "gpt2"})["revision"] is None


def test_prepend_start_token_must_be_true_or_false():
    with pytest.raises(ValueError, match="prepend_start_token"):
        set_params({"model": "gpt2", "prepend_start_token": "yes"})


def test_candidates_are_tried_in_random_order_by_default():
    """Closest-first left more positions short of the threshold on expt-1's
    stimuli (114 of 572 met it, against 140 in random order)."""
    assert set_params({"model": "gpt2"})["candidate_order"] == "random"


def test_seed_defaults_to_0():
    assert set_params({"model": "gpt2"})["seed"] == 0


def test_item_words_are_avoided_by_default():
    assert set_params({"model": "gpt2"})["avoid_item_words"] is True


def test_unknown_candidate_order_raises():
    with pytest.raises(ValueError, match="candidate_order"):
        set_params({"model": "gpt2", "candidate_order": "alphabetical"})


# --- English-only heuristics: on by default for English, off otherwise ---

def test_names_and_abbreviations_are_avoided_by_default_in_english():
    assert set_params({"model": "gpt2"})["avoid_names_and_abbreviations"] is True


def test_names_and_abbreviations_are_not_avoided_by_default_in_other_languages():
    assert set_params({"model": "gpt2", "language": "fr"})["avoid_names_and_abbreviations"] is False


def test_avoiding_names_and_abbreviations_outside_english_raises():
    """The lists are English; asking for them in another language would be
    silently ignored otherwise."""
    with pytest.raises(ValueError, match="avoid_names_and_abbreviations"):
        set_params({"model": "gpt2", "language": "fr", "avoid_names_and_abbreviations": True})


def test_avoid_names_and_abbreviations_must_be_true_or_false():
    with pytest.raises(ValueError, match="avoid_names_and_abbreviations"):
        set_params({"model": "gpt2", "avoid_names_and_abbreviations": "yes"})
