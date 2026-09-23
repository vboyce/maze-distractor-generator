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
    params = {"min_delta": 10, "min_abs": 25, "num_to_test": 20, "language": "fr",
              "include_words": str(french_words), "exclude_words": None}
    out = tmp_path / "out.csv"
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(str(infile), str(out), parameters=params)
    distractors = read_csv(out)[0]["distractors"].split()[1:]
    french = set(wordfreq.top_n_list("fr", 5000))
    assert all(d.strip(".").lower() in french for d in distractors)
