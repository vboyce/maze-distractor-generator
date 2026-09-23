"""The `language` parameter: frequencies and candidate words come from that
language, not always English."""
import sys
import os
import math

import pytest
import wordfreq

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from set_params import set_params
from wordfreq_distractor import get_thresholds, wordfreq_dict, wordfreq_English_dict
from sentence_set import Label

REPO = os.path.dirname(os.path.dirname(__file__))


def ln_freq(word, lang):
    return wordfreq.zipf_frequency(word, lang) * math.log(10)


def test_language_defaults_to_english():
    assert set_params(None)["language"] == "en"


class TestThresholds:
    def test_frequency_bounds_use_the_given_language(self):
        # "grenouille" (frog) is much more frequent in French than in English.
        _, _, min_freq, max_freq = get_thresholds(["grenouille"], {"language": "fr"})
        assert min_freq == pytest.approx(ln_freq("grenouille", "fr") - 1)
        assert max_freq == pytest.approx(ln_freq("grenouille", "fr") + 1)

    def test_english_is_used_without_params(self):
        _, _, min_freq, _ = get_thresholds(["porcupine"])
        assert min_freq == pytest.approx(ln_freq("porcupine", "en") - 1)

    def test_unknown_language_raises(self):
        with pytest.raises(ValueError, match="xx"):
            get_thresholds(["maison"], {"language": "xx"})


class TestDictionary:
    def test_candidates_get_frequencies_from_the_given_language(self, tmp_path):
        words = tmp_path / "fr_words.txt"
        words.write_text("maison\nété\nchat\n")
        d = wordfreq_dict({"language": "fr", "include_words": str(words), "exclude_words": None})
        by_text = {w.text: w.freq for w in d.words}
        assert set(by_text) == {"maison", "été", "chat"}  # accented letters allowed
        assert by_text["maison"] == pytest.approx(math.log(wordfreq.get_frequency_dict("fr")["maison"] * 10**9))

    def test_non_english_language_needs_its_own_word_list(self):
        with pytest.raises(ValueError, match="include_words"):
            wordfreq_dict({"language": "fr"})

    def test_capitalized_and_non_alphabetic_entries_are_dropped(self, tmp_path):
        words = tmp_path / "fr_words.txt"
        words.write_text("maison\nParis\nc'est\n")
        d = wordfreq_dict({"language": "fr", "include_words": str(words), "exclude_words": None})
        assert [w.text for w in d.words] == ["maison"]

    def test_english_dictionary_refuses_another_language(self, monkeypatch):
        monkeypatch.chdir(REPO)
        with pytest.raises(ValueError, match="language"):
            wordfreq_English_dict({"language": "fr"})


def test_label_passes_params_to_the_threshold_function():
    seen = {}

    def threshold_func(words, params):
        seen["params"] = params
        return (2, 10, 1, 20)

    class NoCandidates:
        def get_potential_distractors(self, *args):
            return []

    lab = Label("1", "1")
    lab.add_sentence("maison", "La", 8.0)
    params = {"min_abs": 25, "min_delta": 10, "num_to_test": 10, "language": "fr"}
    lab.choose_distractor(None, NoCandidates(), threshold_func, params, [], "1")
    assert seen["params"]["language"] == "fr"
