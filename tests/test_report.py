"""The per-position report (report_outfile / --report) and the warning for
positions that fell short of their target."""
import sys
import os
import csv
import logging
from unittest.mock import patch

import pytest
import wordfreq

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_stuff
from tests.test_end_to_end import PARAMS, CaseSensitiveBackend, read_csv


def run_with_report(tmp_path, sentence, words, backend=None):
    """Run one item with candidates words; return the report's rows.
    CaseSensitiveBackend: real words score 2 (so the target is 25), "acid"
    40, "actor" 20."""
    infile = tmp_path / "in.csv"
    infile.write_text(f"type,item_num,sentence\nx,1,{sentence}\n")
    word_file = tmp_path / "words.txt"
    word_file.write_text("\n".join(words) + "\n")
    params = {**PARAMS, "include_words": str(word_file)}
    report = tmp_path / "report.csv"
    with patch("main.load_surprisal_model", return_value=backend or CaseSensitiveBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params, report_outfile=str(report))
    return read_csv(report)


def test_the_report_has_a_row_per_position_after_the_first(tmp_path):
    rows = run_with_report(tmp_path, "Birds sing.", ["acid"])
    assert [(r["type"], r["item_num"], r["word"], r["distractor"]) for r in rows] == [("x", "1", "sing.", "acid.")]


def test_a_position_that_met_its_target(tmp_path):
    row = run_with_report(tmp_path, "Birds sing.", ["acid"])[0]
    assert float(row["threshold"]) == 25
    assert float(row["surprisal"]) == 40
    assert row["met"] == "True"


def test_a_position_that_fell_short(tmp_path):
    row = run_with_report(tmp_path, "Birds sing.", ["actor"])[0]
    assert float(row["threshold"]) == 25
    assert float(row["surprisal"]) == 20
    assert row["met"] == "False"


def test_frequencies_are_zipf(tmp_path):
    row = run_with_report(tmp_path, "Birds sing.", ["acid"])[0]
    assert float(row["word_zipf"]) == pytest.approx(wordfreq.zipf_frequency("sing", "en"))
    assert float(row["distractor_zipf"]) == pytest.approx(wordfreq.zipf_frequency("acid", "en"))


class LowerBoundBackend:
    """Batch scoring that gives only a lower bound (the target itself) for
    every candidate; one at a time, a candidate scores 50."""

    model_commit = None

    def get_surprisal(self, prefix, word, base=2.0):
        return 2.0 if word.lower() in {"birds", "sing."} else 50.0

    def score_candidates(self, prefix, candidates, exact_below):
        return [(exact_below, False) for _ in candidates]


def test_the_reported_surprisal_is_exact_when_the_search_had_a_lower_bound(tmp_path):
    row = run_with_report(tmp_path, "Birds sing.", ["acid"], backend=LowerBoundBackend())[0]
    assert float(row["surprisal"]) == 50


def test_one_warning_counts_the_positions_that_fell_short(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        run_with_report(tmp_path, "Birds sing.", ["actor"])
    assert "1 of 1 positions fell short of their target" in caplog.text


def test_a_shared_label_gets_its_own_target_and_surprisal_in_each_sentence():
    """Targets 25 and 40; gap ranking chooses "alpha", 24 after "The" and 45 after "A"."""
    from output import report_rows
    from limit_repeats import Repeatcounter
    from tests.test_sentence_set import (make_two_sentence_set, mock_dict, threshold_func, gap_surprisal,
                                         PARAMS as LABEL_PARAMS)
    ss = make_two_sentence_set()
    with patch("sentence_set.get_surprisal", side_effect=gap_surprisal):
        ss.do_distractors(None, mock_dict(["alpha", "bravo"]), threshold_func, LABEL_PARAMS, Repeatcounter(0))
    rows = report_rows({"1": ss}, "en")
    assert [(r["type"], r["threshold"], r["surprisal"], r["met"]) for r in rows] == [
        ("a", 25, 24.0, False), ("b", 40, 45.0, True)]


def test_in_rejection_mode_only_regenerated_positions_are_reported(tmp_path):
    """Item 1 is not regenerated; in item 2, "sing" keeps its distractor and
    only "loudly." gets a new one."""
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,Birds sing.\ny,2,Birds sing loudly.\n")
    word_file = tmp_path / "words.txt"
    word_file.write_text("acid\nactor\n")
    params = {**PARAMS, "include_words": str(word_file), "max_repeat": 0}
    review = tmp_path / "review.csv"
    with patch("main.load_surprisal_model", return_value=CaseSensitiveBackend()):
        run_stuff(str(infile), str(tmp_path / "out.csv"), parameters=params, longform_outfile=str(review))
        rows = read_csv(review)
        assert [(r["item_num"], r["real_word"]) for r in rows] == [("1", "sing."), ("2", "sing"), ("2", "loudly.")]
        rows[2]["rejected"] = "x"
        with open(review, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = tmp_path / "report.csv"
        run_stuff(str(infile), "", parameters=params, rejection_file=str(review), report_outfile=str(report),
                  longform_outfile=str(tmp_path / "review_2.csv"))
    assert [(r["item_num"], r["word"]) for r in read_csv(report)] == [("2", "loudly.")]
