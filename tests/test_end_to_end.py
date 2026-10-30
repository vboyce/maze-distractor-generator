"""End-to-end runs of run_stuff with a fake surprisal backend (no language model)."""
import sys
import os
import csv
import json
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_stuff

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
    def get_surprisal(self, prefix, word, base=2.0):
        return 2.0 if word in REAL_WORDS else 40.0


PARAMS = {
    "model": "fake",
    "min_delta": 10,
    "min_abs": 25,
    "num_to_test": 20,
    "include_words": os.path.join(REPO, "curated_word_list.txt"),
    "exclude_words": os.path.join(REPO, "exclude.txt"),
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
