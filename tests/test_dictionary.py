import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from wordfreq_distractor import wordfreq_English_dict, get_thresholds

REPO = os.path.dirname(os.path.dirname(__file__))


def test_default_word_list_is_the_curated_list(monkeypatch):
    monkeypatch.chdir(REPO)
    default = wordfreq_English_dict({})
    curated = wordfreq_English_dict({"include_words": "word_lists/curated_word_list.txt"})
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


def test_the_often_capitalized_list_is_loaded_when_checked(tmp_path):
    include = tmp_path / "words.txt"
    include.write_text("angel\n")
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None, "check_often_capitalized": True})
    assert "angel" in d.often_capitalized


def test_the_often_capitalized_list_is_empty_when_not_checked(tmp_path):
    include = tmp_path / "words.txt"
    include.write_text("angel\n")
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None, "check_often_capitalized": False})
    assert d.often_capitalized == set()


def test_the_noun_phrase_breakers_are_loaded_when_used(tmp_path):
    include = tmp_path / "words.txt"
    include.write_text("onto\n")
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None, "break_noun_phrases": True})
    assert "onto" in d.noun_phrase_breakers
    assert "cat" not in d.noun_phrase_breakers


def test_the_noun_phrase_breakers_are_empty_when_not_used(tmp_path):
    include = tmp_path / "words.txt"
    include.write_text("onto\n")
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None, "break_noun_phrases": False})
    assert d.noun_phrase_breakers == set()


def test_only_listed_words_are_candidates_even_beyond_the_cap(tmp_path):
    """With num_to_test 2, the two window words alone would fill the cap;
    "got" is above dog's frequency window and comes in by widening."""
    include = tmp_path / "words.txt"
    include.write_text("cat\ncup\nsky\ngot\n")
    d = wordfreq_English_dict({"include_words": str(include), "exclude_words": None})
    params = {"num_to_test": 2, "candidate_order": "random", "seed": 0}
    window = get_thresholds(["dog"])
    assert d.get_potential_distractors(*window, params, only={"got", "him"}) == ["got"]
    assert set(d.get_potential_distractors(*window, params)) < {"cat", "cup", "sky"}
