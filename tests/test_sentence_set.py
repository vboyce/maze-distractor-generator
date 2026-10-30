import sys
import os
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sentence_set import Label, Sentence, Sentence_Set, item_words
from limit_repeats import Repeatcounter

PARAMS = {"min_abs": 25, "min_delta": 10, "num_to_test": 100, "fallback_ranking": "gap",
          "avoid_item_words": True}


def make_label(word="cat", prefix="The", surprisal=8.0):
    lab = Label("1", 1)
    lab.add_sentence(word, prefix, surprisal)
    return lab


def mock_dict(words):
    d = MagicMock()
    d.get_potential_distractors.return_value = list(words)
    return d


def threshold_func(words, params=None):
    return (2, 10, 1, 20)


def make_sentence_set():
    """Build a minimal Sentence_Set with surprisals pre-set (no model needed)."""
    ss = Sentence_Set("1")
    sent = Sentence(["The", "cat", "sat"], [0, 1, 2], "1", "test")
    sent.surprisal = {1: 8.0, 2: 8.0}
    ss.add(sent)
    ss.make_labels()
    return ss


# --- Label.choose_top_n_distractors ---

class TestChooseTopN:
    def test_returns_n_good_candidates(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast", "jump", "red"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == 3

    def test_candidates_are_from_pool(self):
        lab = make_label()
        pool = ["dog", "run", "fast", "jump", "red"]
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(3, None, mock_dict(pool), threshold_func, PARAMS, [], "1")
        assert all(w in pool for w in result)

    def test_fills_with_fallbacks_if_fewer_good(self):
        lab = make_label()
        # target = max(min_abs=25, surprisal=8+min_delta=10) = 25
        # "dog" returns 30 (>= 25 threshold), others return 5 (< 25)
        def fake_surprisal(backend, prefix, word):
            return 30.0 if word == "dog" else 5.0
        with patch("sentence_set.get_surprisal", side_effect=fake_surprisal):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == 3
        assert result[0] == "dog"  # good candidate first

    def test_respects_banned(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                2, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, ["dog"], "1"
            )
        assert "dog" not in result

    def test_sets_self_distractor_to_first(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert lab.distractor == result[0]

    def test_returns_empty_and_sets_NO_DISTRACTOR_when_no_candidates(self, caplog):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict([]), threshold_func, PARAMS, [], "1"
            )
        assert result == []
        assert lab.distractor == "NO-DISTRACTOR"
        assert "NO-DISTRACTOR" in caplog.text

    def test_no_duplicates_in_result(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1"
            )
        assert len(result) == len(set(result))


# --- Sentence_Set.do_distractors with locked ---

class TestDoDistractorsLocked:
    def test_locked_label_uses_existing_distractor(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        locked = {1: "dog", 2: "run"}
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict([]), threshold_func, PARAMS, repeats, locked=locked)
        assert "dog" in ss.sentences[0].distractor_sentence
        assert "run" in ss.sentences[0].distractor_sentence

    def test_locked_distractor_added_to_repeats(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(1)
        locked = {1: "dog", 2: "run"}
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict([]), threshold_func, PARAMS, repeats, locked=locked)
        assert "dog" in repeats.banned
        assert "run" in repeats.banned

    def test_locked_distractor_in_local_banned_for_subsequent(self):
        """Locked distractor from label 1 must be in banned when label 2 is processed."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        locked = {1: "dog"}
        # Pool only has "dog" — if it's correctly banned, label 2 gets NO-DISTRACTOR
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog"]), threshold_func, PARAMS, repeats, locked=locked)
        # "dog" used for label 1; label 2 can't reuse it, so it gets NO-DISTRACTOR
        words = ss.sentences[0].distractor_sentence.split()
        assert words.count("dog") == 1  # appears only once

    def test_top_n_stores_in_label_options(self):
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        pool = ["dog", "run", "fast", "jump", "red"]
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(pool), threshold_func, PARAMS, repeats, num_options=3)
        for lab in [1, 2]:
            assert lab in ss.label_options
            assert len(ss.label_options[lab]) == 3

    def test_label_options_populated_in_default_mode(self):
        """Even in single-option mode, label_options should be populated."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog", "run"]), threshold_func, PARAMS, repeats)
        assert 1 in ss.label_options
        assert 2 in ss.label_options

    def test_default_behavior_produces_non_empty_distractor_sentence(self):
        """With no locked/num_options, distractor_sentence is set as before."""
        ss = make_sentence_set()
        repeats = Repeatcounter(0)
        with patch("sentence_set.get_surprisal", return_value=30.0):
            ss.do_distractors(None, mock_dict(["dog", "run"]), threshold_func, PARAMS, repeats)
        assert ss.sentences[0].distractor_sentence.startswith("x-x-x")
        assert len(ss.sentences[0].distractor_sentence.split()) == 3


# --- Label.choose_distractor (single best candidate) ---

class TestChooseDistractor:
    def test_returns_first_candidate_that_meets_every_target(self):
        lab = make_label()
        def fake_surprisal(backend, prefix, word):
            return 30.0 if word in ("run", "fast") else 5.0
        with patch("sentence_set.get_surprisal", side_effect=fake_surprisal):
            result = lab.choose_distractor(None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1")
        assert result == "run"
        assert lab.distractor == "run"

    def test_falls_back_to_the_candidate_with_highest_surprisal(self):
        lab = make_label()
        surprisals = {"dog": 5.0, "run": 12.0, "fast": 9.0}
        with patch("sentence_set.get_surprisal", side_effect=lambda backend, prefix, word: surprisals[word]):
            result = lab.choose_distractor(None, mock_dict(["dog", "run", "fast"]), threshold_func, PARAMS, [], "1")
        assert result == "run"

    def test_skips_banned_words_and_the_real_word(self):
        lab = make_label(word="Cat,")
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_distractor(None, mock_dict(["cat", "dog", "run"]), threshold_func, PARAMS, ["dog"], "1")
        assert result == "run"

    def test_returns_NO_DISTRACTOR_when_no_candidates(self):
        """Not x-x-x, which marks the first word, so that the failure shows."""
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=30.0):
            result = lab.choose_distractor(None, mock_dict([]), threshold_func, PARAMS, [], "1")
        assert result == "NO-DISTRACTOR"


# --- Fallback ranking across several sentences with one label ---

def make_two_sentence_set():
    """One item, two sentences sharing label 1, whose targets are 25 and 40."""
    ss = Sentence_Set("1")
    first = Sentence(["The", "cat."], [0, 1], "1", "a")
    first.surprisal = {1: 10.0}  # target max(25, 10 + 10) = 25
    second = Sentence(["A", "cat."], [0, 1], "1", "b")
    second.surprisal = {1: 30.0}  # target max(25, 30 + 10) = 40
    ss.add(first)
    ss.add(second)
    ss.make_labels()
    return ss


# Neither candidate meets both targets.
# "alpha": 26 after "The" (passes), 39 after "A" (fails); lowest overall 26.
# "bravo": 30 after "The" (passes), 35 after "A" (fails); lowest overall 30.
FALLBACK_SURPRISALS = {("The", "alpha"): 26.0, ("A", "alpha"): 39.0,
                       ("The", "bravo"): 30.0, ("A", "bravo"): 35.0}


def fallback_surprisal(backend, prefix, word):
    return FALLBACK_SURPRISALS[(prefix, word)]


MIN_SURPRISAL_PARAMS = {**PARAMS, "fallback_ranking": "min_surprisal"}


class TestMinSurprisalRanking:
    def test_item_gets_the_fallback_with_the_highest_lowest_surprisal(self):
        ss = make_two_sentence_set()
        with patch("sentence_set.get_surprisal", side_effect=fallback_surprisal):
            ss.do_distractors(None, mock_dict(["alpha", "bravo"]), threshold_func, MIN_SURPRISAL_PARAMS,
                              Repeatcounter(0))
        assert [s.distractor_sentence for s in ss.sentences] == ["x-x-x bravo.", "x-x-x bravo."]

    def test_fallbacks_are_ranked_on_their_lowest_surprisal_in_every_sentence(self):
        """Contexts a fallback passes count too, not only the ones it fails."""
        ss = make_two_sentence_set()
        with patch("sentence_set.get_surprisal", side_effect=fallback_surprisal):
            result = ss.labels[1].choose_top_n_distractors(
                2, None, mock_dict(["alpha", "bravo"]), threshold_func, MIN_SURPRISAL_PARAMS, [], "1")
        assert result == ["bravo", "alpha"]


# Neither candidate meets both targets (25 after "The", 40 after "A").
# "alpha": 24 and 45; lowest surprisal 24; worst shortfall 25 - 24 = 1.
# "bravo": 30 and 37; lowest surprisal 30; worst shortfall 40 - 37 = 3.
GAP_SURPRISALS = {("The", "alpha"): 24.0, ("A", "alpha"): 45.0,
                  ("The", "bravo"): 30.0, ("A", "bravo"): 37.0}


def gap_surprisal(backend, prefix, word):
    return GAP_SURPRISALS[(prefix, word)]


def item_distractor(fallback_ranking):
    ss = make_two_sentence_set()
    params = {**PARAMS, "fallback_ranking": fallback_ranking}
    with patch("sentence_set.get_surprisal", side_effect=gap_surprisal):
        ss.do_distractors(None, mock_dict(["alpha", "bravo"]), threshold_func, params, Repeatcounter(0))
    return ss.sentences[0].distractors[1]


class TestFallbackRankingParameter:
    def test_gap_ranking_picks_the_smallest_worst_shortfall(self):
        assert item_distractor("gap") == "alpha."

    def test_min_surprisal_ranking_picks_the_highest_lowest_surprisal(self):
        assert item_distractor("min_surprisal") == "bravo."

    def test_gap_ties_go_to_the_earliest_candidate(self):
        lab = make_label()  # one context, target 25
        with patch("sentence_set.get_surprisal", return_value=20.0):
            result = lab.choose_top_n_distractors(
                3, None, mock_dict(["dog", "run", "fast"]), threshold_func,
                {**PARAMS, "fallback_ranking": "gap"}, [], "1")
        assert result == ["dog", "run", "fast"]

    def test_unknown_ranking_raises(self):
        lab = make_label()
        with patch("sentence_set.get_surprisal", return_value=20.0):
            with pytest.raises(ValueError, match="fallback_ranking"):
                lab.choose_top_n_distractors(
                    1, None, mock_dict(["dog"]), threshold_func,
                    {**PARAMS, "fallback_ranking": "closest"}, [], "1")


# --- Words of the item (avoid_item_words) ---

class TestItemWords:
    def test_words_are_lowercased_without_punctuation(self):
        assert item_words(["The", "Cat,", "sat."]) == {"the", "cat", "sat"}

    def test_possessives_and_hyphenated_words_are_split(self):
        assert item_words(["cat's", "well-known"]) == {"cat", "s", "well", "known"}

    def test_a_word_from_another_sentence_of_the_item_is_not_chosen(self):
        ss = make_two_sentence_set()  # "The cat." and "A cat."
        with patch("sentence_set.get_surprisal", return_value=40.0):
            ss.do_distractors(None, mock_dict(["the", "a", "dog"]), threshold_func,
                              {**PARAMS, "avoid_item_words": True}, Repeatcounter(0))
        assert ss.labels[1].distractor == "dog"

    def test_off_only_the_label_s_own_words_are_avoided(self):
        ss = make_two_sentence_set()
        with patch("sentence_set.get_surprisal", return_value=40.0):
            ss.do_distractors(None, mock_dict(["the", "a", "dog"]), threshold_func,
                              {**PARAMS, "avoid_item_words": False}, Repeatcounter(0))
        assert ss.labels[1].distractor == "the"
