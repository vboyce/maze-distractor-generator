import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wordfreq_distractor import wordfreq_English_dict

REPO = os.path.dirname(os.path.dirname(__file__))


def test_default_word_list_is_the_curated_list(monkeypatch):
    monkeypatch.chdir(REPO)
    default = wordfreq_English_dict({})
    curated = wordfreq_English_dict({"include_words": "curated_word_list.txt"})
    assert sorted(w.text for w in default.words) == sorted(w.text for w in curated.words)
