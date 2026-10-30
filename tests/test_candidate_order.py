"""The order candidate distractors are tried in (candidate_order, seed)."""
import sys
import os
import math
import random

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wordfreq_distractor import wordfreq_dict, get_frequency

WORDS = ["horse", "table", "chair", "apple", "mauve", "tulip", "cat", "dog", "garden",
         "pencil", "kitchen", "blanket", "rose", "lamp"]
TARGET = ["house."]  # length 5 once its punctuation is stripped
WINDOW = (3, 7, 0, 30)  # lengths 3-7, any frequency: every word above


def make_dict(tmp_path, words=WORDS, **params):
    include = tmp_path / "words.txt"
    include.write_text("\n".join(words))
    return wordfreq_dict({"include_words": str(include), "exclude_words": None, "seed": 0,
                          "candidate_order": "closest", "num_to_test": 100, **params})


def order(d, seed=0, candidate_order="closest", order_key="1\0label"):
    params = {"num_to_test": 100, "seed": seed, "candidate_order": candidate_order}
    return d.get_potential_distractors(*WINDOW, params, target_words=TARGET, order_key=order_key)


def frequency_tier(word):
    distance = abs(get_frequency(word) - get_frequency("house"))
    return math.floor(distance / math.log(2))


class TestClosestOrder:
    def test_closer_lengths_come_first(self, tmp_path):
        length_gaps = [abs(len(w) - 5) for w in order(make_dict(tmp_path))]
        assert length_gaps == sorted(length_gaps)

    def test_within_a_length_closer_frequencies_come_first(self, tmp_path):
        five_letters = [w for w in order(make_dict(tmp_path)) if len(w) == 5]
        tiers = [frequency_tier(w) for w in five_letters]
        assert tiers == sorted(tiers)
        assert len(set(tiers)) > 1  # the test would pass trivially otherwise

    def test_same_seed_same_order(self, tmp_path):
        assert order(make_dict(tmp_path), seed=1) == order(make_dict(tmp_path), seed=1)

    def test_different_seed_different_order(self, tmp_path):
        assert order(make_dict(tmp_path), seed=1) != order(make_dict(tmp_path), seed=2)

    def test_different_position_different_order(self, tmp_path):
        d = make_dict(tmp_path)
        assert order(d, order_key="1\0a") != order(d, order_key="1\0b")

    def test_excluding_a_word_leaves_the_order_of_the_rest_unchanged(self, tmp_path):
        full = order(make_dict(tmp_path))
        without = order(make_dict(tmp_path, words=[w for w in WORDS if w != "chair"]))
        assert without == [w for w in full if w != "chair"]

    def test_num_to_test_keeps_the_closest(self, tmp_path):
        d = make_dict(tmp_path)
        params = {"num_to_test": 4, "seed": 0, "candidate_order": "closest"}
        closest = d.get_potential_distractors(*WINDOW, params, target_words=TARGET, order_key="1\0label")
        assert closest == order(d)[:4]


class TestRandomOrder:
    def test_is_the_window_shuffled_by_a_generator_seeded_with_seed(self, tmp_path):
        """The same shuffle as before candidate_order existed, after random.seed(seed)."""
        d = make_dict(tmp_path, seed=5, candidate_order="random")
        expected = d.get_words(*WINDOW)
        random.Random(5).shuffle(expected)
        assert order(d, seed=5, candidate_order="random") == expected
