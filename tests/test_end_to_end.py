"""End-to-end runs of run_stuff with a fake surprisal backend (no language model)."""
import sys
import os
import csv
import json
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_stuff
from wordfreq_distractor import wordfreq_dict
from utils import strip_punct

REPO = os.path.dirname(os.path.dirname(__file__))

INPUT = """\
type,item_num,sentence,labels
sub_rel,3,"The cat who the dog scared hid.",pre1 pre2 who art noun verb main
obj_rel,3,"The dog who scared the cat hid.",pre1 pre2 who verb art noun main
filler,4,"Birds sing in the morning.",
"""

# Real words are unsurprising and every other word is surprising, so the first
# candidate for each position always meets the threshold.
REAL_WORDS = {"cat", "who", "the", "dog", "scared", "hid.", "sing", "in", "morning."}


class FakeBackend:
    model_commit = None

    def get_surprisal(self, prefix, word, base=2.0):
        return 2.0 if word in REAL_WORDS else 40.0


PARAMS = {
    "model": "fake",
    "min_delta": 10,
    "min_abs": 25,
    "num_to_test": 20,
    "include_words": os.path.join(REPO, "word_lists", "curated_word_list.txt"),
    "exclude_words": os.path.join(REPO, "word_lists", "exclude.txt"),
    "max_repeat": 1,
}


@pytest.fixture
def infile(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text(INPUT)
    return str(f)


def run(infile, outfile, **kwargs):
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, outfile, parameters=PARAMS, **kwargs)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_delim_output_has_one_distractor_per_word(infile, tmp_path):
    out = tmp_path / "out.csv"
    run(infile, str(out))
    rows = read_csv(out)
    assert [r["type"] for r in rows] == ["sub_rel", "obj_rel", "filler"]
    for row in rows:
        distractors = row["distractors"].split()
        assert len(distractors) == len(row["sentence"].split())
        assert distractors[0] == "x-x-x"


def test_distractors_copy_final_punctuation(infile, tmp_path):
    out = tmp_path / "out.csv"
    run(infile, str(out))
    for row in read_csv(out):
        assert row["distractors"].endswith(".")


def test_shared_labels_get_the_same_distractor(infile, tmp_path):
    out = tmp_path / "out.csv"
    run(infile, str(out))
    sub, obj = read_csv(out)[:2]
    sub_by_label = dict(zip(sub["labels"].split(), sub["distractors"].split()))
    obj_by_label = dict(zip(obj["labels"].split(), obj["distractors"].split()))
    for label in ["who", "verb", "art", "noun", "main"]:
        assert sub_by_label[label] == obj_by_label[label]


def test_no_distractor_is_used_twice_with_max_repeat_1(infile, tmp_path):
    out = tmp_path / "out.csv"
    run(infile, str(out))
    rows = read_csv(out)
    per_item = {}
    for row in rows:
        per_item.setdefault(row["item_num"], set()).update(row["distractors"].lower().strip(".").split()[1:])
    words = [w for item_words in per_item.values() for w in item_words]
    assert len(words) == len(set(words))


def test_json_output_is_a_js_module(infile, tmp_path):
    out = tmp_path / "stim.js"
    run(infile, str(out), outformat="json", module_name="items")
    text = out.read_text()
    items = json.loads(text[len("export const items = "):].rstrip().rstrip(";"))
    assert len(items) == 3
    assert items[2]["sent"] == "Birds sing in the morning."


def test_rejection_round_trip_with_string_labels(infile, tmp_path):
    review = tmp_path / "review.csv"
    run(infile, str(tmp_path / "out.csv"), longform_outfile=str(review))
    rows = read_csv(review)
    # reject the distractor for label "noun" in item 3 (one of its two rows)
    target = next(i for i, r in enumerate(rows) if r["item_num"] == "3" and r["label"] == "noun")
    rows[target]["rejected"] = "yes"
    with open(review, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    updated = tmp_path / "updated.csv"
    run(infile, "", rejection_file=str(review), longform_outfile=str(updated))
    new_rows = read_csv(updated)

    assert all(r["rejected"] == "" for r in new_rows)
    unchanged = [i for i, r in enumerate(rows) if not (r["item_num"] == "3" and r["label"] == "noun")]
    for i in unchanged:
        assert new_rows[i]["distractor"] == rows[i]["distractor"]


def test_rejection_mode_output_has_every_sentence_with_final_distractors(infile, tmp_path):
    review = tmp_path / "review.csv"
    run(infile, str(tmp_path / "out.csv"), longform_outfile=str(review))
    rows = read_csv(review)
    rows[0]["rejected"] = "x"
    with open(review, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    updated = tmp_path / "updated.csv"
    final = tmp_path / "final.csv"
    run(infile, str(final), rejection_file=str(review), longform_outfile=str(updated))

    final_rows = read_csv(final)
    assert [r["type"] for r in final_rows] == ["sub_rel", "obj_rel", "filler"]
    # Rebuild each sentence's distractors from the updated review file and compare.
    by_position = {(r["item_num"], r["label"], r["prefix"]): r["distractor"] for r in read_csv(updated)}
    for row in final_rows:
        words, labels = row["sentence"].split(), row["labels"].split()
        expected = ["x-x-x"] + [by_position[(row["item_num"], labels[i], " ".join(words[:i]))]
                                for i in range(1, len(words))]
        assert row["distractors"] == " ".join(expected)


def test_rejection_mode_with_nothing_rejected_converts_the_review_file(infile, tmp_path):
    first = tmp_path / "out.csv"
    review = tmp_path / "review.csv"
    run(infile, str(first), longform_outfile=str(review))
    converted = tmp_path / "converted.csv"
    run(infile, str(converted), rejection_file=str(review))
    assert read_csv(converted) == read_csv(first)


def test_non_english_run_draws_distractors_from_that_language(tmp_path):
    import wordfreq
    french_words = tmp_path / "fr_words.txt"
    french_words.write_text("\n".join(wordfreq.top_n_list("fr", 5000)))
    infile = tmp_path / "fr.csv"
    infile.write_text("type,item_num,sentence\nx,1,Le chat dort sur le canapé.\n")
    params = {"model": "fake", "min_delta": 10, "min_abs": 25, "num_to_test": 20, "language": "fr",
              "include_words": str(french_words), "exclude_words": None}
    out = tmp_path / "out.csv"
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(str(infile), str(out), parameters=params)
    distractors = read_csv(out)[0]["distractors"].split()[1:]
    french = set(wordfreq.top_n_list("fr", 5000))
    assert all(d.strip(".").lower() in french for d in distractors)


def test_run_without_a_model_raises_before_loading_one(infile, tmp_path):
    params = {key: value for key, value in PARAMS.items() if key != "model"}
    with patch("main.load_surprisal_model") as load:
        with pytest.raises(ValueError, match="model"):
            run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    load.assert_not_called()


def test_model_override_supplies_the_required_model(infile, tmp_path):
    params = {key: value for key, value in PARAMS.items() if key != "model"}
    with patch("main.load_surprisal_model", return_value=FakeBackend()) as load:
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params, model_override="distilgpt2")
    assert load.call_args.args[0] == "distilgpt2"


def test_run_passes_start_token_and_revision_to_the_backend(infile, tmp_path):
    params = {**PARAMS, "prepend_start_token": False, "revision": "abc123"}
    with patch("main.load_surprisal_model", return_value=FakeBackend()) as load:
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    assert load.call_args.kwargs["prepend_start_token"] is False
    assert load.call_args.kwargs["revision"] == "abc123"


def distractor_rows(infile, tmp_path, name, **params):
    out = tmp_path / name
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, str(out), parameters={**PARAMS, **params})
    return [row["distractors"] for row in read_csv(out)]


@pytest.mark.parametrize("candidate_order", ["closest", "random"])
def test_same_seed_gives_the_same_distractors(infile, tmp_path, candidate_order):
    first = distractor_rows(infile, tmp_path, "a.csv", candidate_order=candidate_order, seed=3)
    second = distractor_rows(infile, tmp_path, "b.csv", candidate_order=candidate_order, seed=3)
    assert first == second


@pytest.mark.parametrize("candidate_order", ["closest", "random"])
def test_different_seeds_give_different_distractors(infile, tmp_path, candidate_order):
    first = distractor_rows(infile, tmp_path, "a.csv", candidate_order=candidate_order, seed=3)
    second = distractor_rows(infile, tmp_path, "b.csv", candidate_order=candidate_order, seed=4)
    assert first != second


def test_no_word_of_the_item_is_a_distractor(tmp_path):
    # Every candidate scores 40 bits, so without avoid_item_words the first
    # candidate in the pool would be chosen; here that is a word of the item.
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,The tiger's well-known stripes.\nx,2,Birds sing.\n")
    pool = ["stripes", "tiger", "known", "dog", "cat", "hen", "owl", "elk"]
    params = {**PARAMS, "max_repeat": 0}
    with patch("main.load_surprisal_model", return_value=FakeBackend()), \
            patch("wordfreq_distractor.wordfreq_dict.get_potential_distractors", return_value=pool):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params)
    distractors = read_csv(tmp_path / "out.csv")[0]["distractors"].split()[1:]
    assert not {"stripes", "tiger", "known", "well"} & {d.strip(".").lower() for d in distractors}


def test_position_with_no_candidates_gets_NO_DISTRACTOR_and_a_warning(tmp_path, caplog):
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,Birds sing.\n")
    with patch("main.load_surprisal_model", return_value=FakeBackend()), \
            patch("wordfreq_distractor.wordfreq_dict.get_potential_distractors", return_value=[]):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=PARAMS)
    assert read_csv(tmp_path / "out.csv")[0]["distractors"] == "x-x-x NO-DISTRACTOR."
    assert "NO-DISTRACTOR" in caplog.text


RUN_IN_SUBPROCESS = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from main import run_stuff
from wordfreq_distractor import wordfreq_dict
from tests.test_end_to_end import FakeBackend, PARAMS
with patch("main.load_surprisal_model", return_value=FakeBackend()):
    run_stuff(sys.argv[2], sys.argv[3], parameters={**PARAMS, "candidate_order": "random", "seed": 1})
"""


def test_runs_in_different_processes_give_the_same_distractors(infile, tmp_path):
    """Python randomizes string hashing per process, so nothing may depend
    on the iteration order of a set of strings."""
    import subprocess
    outputs = []
    for hash_seed in ["1", "2"]:
        out = tmp_path / f"out_{hash_seed}.csv"
        subprocess.run([sys.executable, "-c", RUN_IN_SUBPROCESS, REPO, infile, str(out)], check=True,
                       cwd=REPO, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        outputs.append(read_csv(out))
    assert outputs[0] == outputs[1]


def test_used_up_candidates_do_not_count_toward_num_to_test(tmp_path):
    """Banned words (used max_repeat times, or words of the item) are dropped
    before candidates are capped at num_to_test, so the search widens instead
    of giving up. With closest-first order the same few words head the list
    at every short, common target word, so late items ran out."""
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,Birds sing.\nx,2,Birds sing.\n")
    # "ring" is in sing's frequency window; "sings" is just below it.
    words = tmp_path / "words.txt"
    words.write_text("ring\nsings\n")
    params = {**PARAMS, "include_words": str(words), "num_to_test": 1, "max_repeat": 1}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params)
    assert [r["distractors"] for r in read_csv(tmp_path / "out.csv")] == ["x-x-x ring.", "x-x-x sings."]


def test_first_names_are_not_used_as_distractors_by_default(tmp_path):
    """Readers know "adam" and "alan" only as names, so they would read as
    plausible continuations. Only the names are in sing's frequency window;
    "sings", just below it, comes in when the search widens."""
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,Birds sing.\n")
    words = tmp_path / "words.txt"
    words.write_text("adam\nalan\nsings\n")
    params = {**PARAMS, "include_words": str(words)}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params)
    assert read_csv(tmp_path / "out.csv")[0]["distractors"] == "x-x-x sings."


class CaseSensitiveBackend:
    """Real words are unsurprising. "acid" and "angel" meet the target (25)
    in lowercase but not capitalized; "actor" falls 5 short in either case."""

    model_commit = None

    def get_surprisal(self, prefix, word, base=2.0):
        if word.lower() in {"birds", "sing", "sing."}:
            return 2.0
        if word in {"acid", "angel"}:
            return 40.0
        if word in {"Acid", "Angel"}:
            return 2.0
        return 20.0


def run_with_words(tmp_path, sentence, words):
    infile = tmp_path / "in.csv"
    infile.write_text(f"type,item_num,sentence\nx,1,{sentence}\n")
    word_file = tmp_path / "words.txt"
    word_file.write_text("\n".join(words) + "\n")
    params = {**PARAMS, "include_words": str(word_file)}
    with patch("main.load_surprisal_model", return_value=CaseSensitiveBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params)
    return read_csv(tmp_path / "out.csv")[0]["distractors"]


def test_a_distractor_shown_capitalized_must_meet_the_target_capitalized(tmp_path):
    """"Sing" is capitalized, so its distractor is shown capitalized. "Acid"
    misses the target by 23 bits; "Actor" by 5, so it is the better fallback."""
    assert run_with_words(tmp_path, "Birds Sing.", ["acid", "actor"]) == "x-x-x Actor."


def test_an_often_capitalized_word_must_meet_the_target_capitalized(tmp_path):
    """A reader may take "angel" for the name Angel, so it is scored
    capitalized too, though shown in lowercase."""
    assert run_with_words(tmp_path, "Birds sing.", ["angel", "actor"]) == "x-x-x actor."


# "cat", "cup" and "sky" are nouns in dog's length and frequency window; "got",
# on noun_phrase_breakers.txt, is more frequent, so it comes in only when the
# search widens.
NOUNS_AND_A_BREAKER = ["cat", "cup", "sky", "got"]


def distractor_for_dog(tmp_path, sentences, **params):
    """The distractor at "dog" in the first of sentences (one item), with
    every non-real word meeting the target."""
    words = sentences[0].split()
    dog = [strip_punct(word).lower() for word in words].index("dog")
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\n" + "".join(f"x,1,{s}\n" for s in sentences))
    word_file = tmp_path / "words.txt"
    word_file.write_text("\n".join(NOUNS_AND_A_BREAKER) + "\n")
    params = {**PARAMS, "include_words": str(word_file), "candidate_order": "random", **params}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params)
    return strip_punct(read_csv(tmp_path / "out.csv")[0]["distractors"].split()[dog]).lower()


def test_after_a_determiner_only_noun_phrase_breakers_are_tried(tmp_path):
    """"The cat" could go on as a sentence however surprising "cat" is."""
    assert distractor_for_dog(tmp_path, ["The dog hid."]) == "got"


def test_after_another_word_any_candidate_is_tried(tmp_path):
    """Words in the window are tried first; "got" only after them."""
    assert distractor_for_dog(tmp_path, ["Big dog hid."]) in {"cat", "cup", "sky"}


def test_noun_phrase_breakers_are_not_required_when_turned_off(tmp_path):
    assert distractor_for_dog(tmp_path, ["The dog hid."], break_noun_phrases=False) in {"cat", "cup", "sky"}


def test_a_determiner_ending_a_sentence_does_not_restrict_the_next_word(tmp_path):
    """"his" ends the sentence: it is a pronoun there, and "Dog" starts a
    new noun phrase."""
    assert distractor_for_dog(tmp_path, ["It was his. Dog hid."]) in {"cat", "cup", "sky"}


def test_a_determiner_after_opening_punctuation_still_counts(tmp_path):
    assert distractor_for_dog(tmp_path, ["(The dog hid.)"]) == "got"


def test_a_label_after_a_determiner_in_any_sentence_gets_a_noun_phrase_breaker(tmp_path):
    assert distractor_for_dog(tmp_path, ["Big dog hid.", "The dog hid."]) == "got"


class DictionaryWithoutOftenCapitalized(wordfreq_dict):
    """A dictionary class written before check_often_capitalized existed."""

    def __init__(self, params):
        super().__init__(params)
        del self.often_capitalized


OLD_DICTIONARY = {"dictionary_loc": "tests.test_end_to_end", "dictionary_class": "DictionaryWithoutOftenCapitalized"}


def test_a_dictionary_without_often_capitalized_raises_before_loading_the_model(infile, tmp_path):
    params = {**PARAMS, **OLD_DICTIONARY}
    with patch("main.load_surprisal_model") as load:
        with pytest.raises(ValueError, match="check_often_capitalized"):
            run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    load.assert_not_called()


def test_a_dictionary_without_often_capitalized_works_when_not_checked(infile, tmp_path):
    params = {**PARAMS, **OLD_DICTIONARY, "check_often_capitalized": False}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    assert len(read_csv(tmp_path / "out.csv")) == 3


class DictionaryWithoutNounPhraseBreakers(wordfreq_dict):
    """A dictionary class written before break_noun_phrases existed."""

    def __init__(self, params):
        super().__init__(params)
        del self.noun_phrase_breakers


DICTIONARY_WITHOUT_BREAKERS = {"dictionary_loc": "tests.test_end_to_end",
                               "dictionary_class": "DictionaryWithoutNounPhraseBreakers"}


def test_a_dictionary_without_noun_phrase_breakers_raises_before_loading_the_model(infile, tmp_path):
    params = {**PARAMS, **DICTIONARY_WITHOUT_BREAKERS}
    with patch("main.load_surprisal_model") as load:
        with pytest.raises(ValueError, match="break_noun_phrases"):
            run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    load.assert_not_called()


def test_a_dictionary_without_noun_phrase_breakers_works_when_not_used(infile, tmp_path):
    params = {**PARAMS, **DICTIONARY_WITHOUT_BREAKERS, "break_noun_phrases": False}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    assert len(read_csv(tmp_path / "out.csv")) == 3


class DictionaryWithOldSignature(DictionaryWithoutNounPhraseBreakers):
    """get_potential_distractors as written before break_noun_phrases existed."""

    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params,
                                  target_words=None, order_key=None, avoid=()):
        return super().get_potential_distractors(min_length, max_length, min_freq, max_freq, params,
                                                 target_words=target_words, order_key=order_key, avoid=avoid)


def test_a_dictionary_with_the_old_signature_works_when_noun_phrases_are_not_broken(infile, tmp_path):
    params = {**PARAMS, "dictionary_loc": "tests.test_end_to_end", "dictionary_class": "DictionaryWithOldSignature",
              "break_noun_phrases": False}
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    assert len(read_csv(tmp_path / "out.csv")) == 3


def test_the_default_word_lists_are_found_from_any_directory(infile, tmp_path, monkeypatch):
    """include_words and exclude_words left out: the default lists come with
    the generator, so a run started elsewhere still finds them."""
    params = {key: value for key, value in PARAMS.items() if key not in ("include_words", "exclude_words")}
    monkeypatch.chdir(tmp_path)
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, str(tmp_path / "out.csv"), parameters=params)
    assert len(read_csv(tmp_path / "out.csv")) == 3
