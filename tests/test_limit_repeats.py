import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from limit_repeats import Repeatcounter


def test_word_is_banned_once_it_reaches_max():
    repeats = Repeatcounter(2)
    repeats.increment("dog")
    assert "dog" not in repeats.banned
    repeats.increment("dog")
    assert repeats.banned == ["dog"]


def test_banned_list_has_no_duplicates_when_word_is_used_again():
    repeats = Repeatcounter(1)
    repeats.increment("dog")
    repeats.increment("dog")
    assert repeats.banned == ["dog"]


def test_max_zero_means_no_limit():
    repeats = Repeatcounter(0)
    for _ in range(5):
        repeats.increment("dog")
    assert repeats.banned == []
