"""Batch scoring of candidate distractors on the causal backend gives the
same choices as scoring one candidate at a time."""
import sys
import os
import itertools

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backends.transformers_backend import CausalLMBackend
from limit_repeats import Repeatcounter
from sentence_set import Sentence, Sentence_Set
from tests.test_sentence_set import mock_dict, threshold_func

SINGLE = ["cat", "dog", "sat", "ran", "sat.", "ran.", "A", "The"]
# Two-token "words", so that some candidates need more than their first token scored.
POOL = SINGLE + [f"{a} {b}" for a, b in itertools.product(SINGLE[:4], SINGLE[:4])]


class OneAtATime:
    """The same model, with only get_surprisal: the slow path."""

    def __init__(self, backend):
        self.backend = backend

    def get_surprisal(self, prefix, word, base=2.0):
        return self.backend.get_surprisal(prefix, word, base)


@pytest.fixture(scope="module")
def fast_backend(tiny_lm_dir):
    return CausalLMBackend(tiny_lm_dir)


LOWERCASE_ITEM = [["The", "cat", "sat."], ["A", "dog", "ran."]]
# Capitalized real words, so each candidate is also scored capitalized.
CAPITALIZED_ITEM = [["The", "Cat", "sat."], ["A", "Dog", "Ran."]]


def run_item(backend, fallback_ranking, num_options, sentences=LOWERCASE_ITEM, often_capitalized=()):
    """Distractors for a two-sentence item whose labels are shared; the tiny
    model's surprisals are around 3.3 bits a token, so with these targets
    some candidates pass and some don't."""
    ss = Sentence_Set("1")
    for words, tag in zip(sentences, ["a", "b"]):
        ss.add(Sentence(words, [0, 1, 2], "1", tag))
    ss.do_surprisals(backend)
    ss.make_labels()
    # The pool is the tiny vocabulary, which is also the item's words, so
    # avoid_item_words is off.
    params = {"min_abs": 3.4, "min_delta": 0, "num_to_test": len(POOL), "fallback_ranking": fallback_ranking,
              "avoid_item_words": False, "check_shown_case": True,
              "check_often_capitalized": bool(often_capitalized)}
    dictionary = mock_dict(POOL)
    dictionary.often_capitalized = set(often_capitalized)
    ss.do_distractors(backend, dictionary, threshold_func, params, Repeatcounter(0), num_options=num_options)
    return ss.label_options, [s.distractor_sentence for s in ss.sentences]


@pytest.mark.parametrize("fallback_ranking", ["gap", "min_surprisal"])
@pytest.mark.parametrize("num_options", [1, 5])
def test_batch_scoring_chooses_what_one_at_a_time_chooses(fast_backend, fallback_ranking, num_options):
    fast = run_item(fast_backend, fallback_ranking, num_options)
    slow = run_item(OneAtATime(fast_backend), fallback_ranking, num_options)
    assert fast == slow


@pytest.mark.parametrize("fallback_ranking", ["gap", "min_surprisal"])
def test_batch_scoring_chooses_what_one_at_a_time_chooses_with_capitalized_forms(fast_backend, fallback_ranking):
    """Each chunk mixes lowercase and capitalized forms of its candidates."""
    item = {"sentences": CAPITALIZED_ITEM, "often_capitalized": ["dog", "sat"]}
    fast = run_item(fast_backend, fallback_ranking, 5, **item)
    slow = run_item(OneAtATime(fast_backend), fallback_ranking, 5, **item)
    assert fast == slow


class ForwardCounter:
    """Counts the model's forward passes, with a forward hook."""

    def __init__(self, model):
        self.count = 0
        self.handle = model.register_forward_hook(self.hook)

    def hook(self, module, inputs, outputs):
        self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def test_batch_scoring_needs_far_fewer_forward_passes(fast_backend):
    with ForwardCounter(fast_backend.model) as slow:
        run_item(OneAtATime(fast_backend), "gap", 5)
    with ForwardCounter(fast_backend.model) as fast:
        run_item(fast_backend, "gap", 5)
    assert fast.count * 4 < slow.count


class TestScoreCandidates:
    def test_exact_scores_equal_one_at_a_time_scores(self, fast_backend):
        scores = fast_backend.score_candidates("The cat", POOL, exact_below=float("inf"))
        for word, (surprisal, exact) in zip(POOL, scores):
            assert exact
            assert surprisal == pytest.approx(fast_backend.get_surprisal("The cat", word), abs=1e-4)

    def test_single_token_scores_are_exact_whatever_the_bound(self, fast_backend):
        scores = fast_backend.score_candidates("The", ["dog", "sat."], exact_below=0.0)
        assert [exact for _, exact in scores] == [True, True]
        assert scores[0][0] == pytest.approx(fast_backend.get_surprisal("The", "dog"), abs=1e-4)

    def test_multi_token_word_whose_first_token_meets_the_bound_gets_that_lower_bound(self, fast_backend):
        (surprisal, exact), = fast_backend.score_candidates("The", ["dog ran"], exact_below=0.0)
        assert not exact
        assert surprisal == pytest.approx(fast_backend.get_surprisal("The", "dog"), abs=1e-4)
        assert surprisal < fast_backend.get_surprisal("The", "dog ran")

    def test_text_longer_than_the_model_reads_raises(self, fast_backend):
        from tests.tiny_lm import MAX_POSITIONS
        with pytest.raises(ValueError, match="longer than the model reads"):
            fast_backend.score_candidates(" ".join(["cat"] * MAX_POSITIONS), ["dog"], exact_below=0.0)


class CountingBackend:
    """Fake fast-path backend: every candidate scores 40 bits, exactly."""

    def __init__(self):
        self.scored = []

    def score_candidates(self, prefix, candidates, exact_below):
        self.scored.extend(candidates)
        return [(40.0, True)] * len(candidates)


def test_search_stops_after_the_chunk_that_finds_enough_good_candidates():
    from tests.test_sentence_set import make_label, PARAMS
    backend = CountingBackend()
    pool = [f"w{i}" for i in range(100)]
    result = make_label().choose_top_n_distractors(1, backend, mock_dict(pool), threshold_func, PARAMS, [], "1")
    assert result == ["w0"]
    assert backend.scored == pool[:16]


def test_later_chunks_double_in_size():
    from tests.test_sentence_set import make_label, PARAMS
    backend = CountingBackend()
    pool = [f"w{i}" for i in range(100)]
    make_label().choose_top_n_distractors(17, backend, mock_dict(pool), threshold_func, PARAMS, [], "1")
    assert backend.scored == pool[:16 + 32]
