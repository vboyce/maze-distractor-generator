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


def dictionary_words(tmp_path, words, **params):
    include = tmp_path / "words.txt"
    include.write_text("\n".join(words))
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None, **params})
    return {w.text for w in d.words}


def test_avoiding_names_and_abbreviations_leaves_out_listed_words(tmp_path):
    """"adam" is on first_names.txt, "co" on abbreviations.txt."""
    assert dictionary_words(tmp_path, ["adam", "co", "sings"], avoid_names_and_abbreviations=True) == {"sings"}


def test_names_and_abbreviations_stay_when_not_avoided(tmp_path):
    assert dictionary_words(tmp_path, ["adam", "co", "sings"], avoid_names_and_abbreviations=False) == {
        "adam", "co", "sings"}
