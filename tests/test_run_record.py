"""The JSON run record written beside the output of every run."""
import sys
import os
import json
import hashlib
import subprocess
from importlib import metadata
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_stuff
from set_params import set_params
from tests.test_end_to_end import PARAMS, FakeBackend, INPUT

REPO = os.path.dirname(os.path.dirname(__file__))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def infile(tmp_path):
    f = tmp_path / "in.csv"
    f.write_text(INPUT)
    return str(f)


def run(infile, outfile, **kwargs):
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        run_stuff(infile, outfile, parameters=PARAMS, **kwargs)


def test_the_record_is_written_beside_the_output(infile, tmp_path):
    run(infile, str(tmp_path / "out.csv"))
    assert (tmp_path / "out.csv.run.json").exists()


def test_the_record_can_be_written_elsewhere(infile, tmp_path):
    run(infile, str(tmp_path / "out.csv"), run_record_outfile=str(tmp_path / "record.json"))
    assert (tmp_path / "record.json").exists()
    assert not (tmp_path / "out.csv.run.json").exists()


def read_record(infile, tmp_path):
    run(infile, str(tmp_path / "out.csv"))
    with open(tmp_path / "out.csv.run.json") as f:
        return json.load(f)


def test_the_record_has_every_parameter(infile, tmp_path):
    assert read_record(infile, tmp_path)["parameters"] == set_params(PARAMS)


def test_the_record_names_the_model(infile, tmp_path):
    """FakeBackend, like a model loaded from a local directory, has no commit."""
    assert read_record(infile, tmp_path)["model"] == {"name": "fake", "backend": "transformers",
                                                       "revision": None, "commit": None}


def test_the_record_has_the_package_versions(infile, tmp_path):
    versions = read_record(infile, tmp_path)["versions"]
    for package in ("torch", "transformers", "wordfreq"):
        assert versions[package] == metadata.version(package)


def test_the_record_has_the_generator_commit(infile, tmp_path):
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    assert read_record(infile, tmp_path)["generator"]["commit"] == head.stdout.strip()


def test_the_record_has_checksums_of_the_input_and_word_lists(infile, tmp_path):
    files = read_record(infile, tmp_path)["files"]
    assert files["input"] == {"path": infile, "sha256": sha256(infile)}
    listed = {entry["path"]: entry["sha256"] for entry in files["word_lists"]}
    assert listed[PARAMS["include_words"]] == sha256(PARAMS["include_words"])
    assert listed[PARAMS["exclude_words"]] == sha256(PARAMS["exclude_words"])
    # the English heuristics' lists, all on by default
    for name in ("first_names.txt", "abbreviations.txt", "often_capitalized.txt", "noun_phrase_breakers.txt"):
        path = os.path.join(REPO, "word_lists", name)
        assert listed[path] == sha256(path)


def test_a_parameter_json_cannot_hold_is_an_error(infile, tmp_path):
    """inf would be written as Infinity, which is not JSON."""
    with patch("main.load_surprisal_model", return_value=FakeBackend()):
        with pytest.raises(ValueError, match="run record can't hold"):
            run_stuff(infile, str(tmp_path / "out.csv"), parameters={**PARAMS, "min_abs": float("inf")})
    assert not (tmp_path / "out.csv").exists()  # failed before the run


def test_in_rejection_mode_with_no_output_the_record_goes_beside_the_longform(infile, tmp_path):
    out, review = tmp_path / "out.csv", tmp_path / "review.csv"
    run(infile, str(out), longform_outfile=str(review))
    run(infile, "", rejection_file=str(review), longform_outfile=str(tmp_path / "review_2.csv"))
    assert (tmp_path / "review_2.csv.run.json").exists()


def test_a_run_with_nowhere_to_put_the_record_fails_before_loading_the_model(infile, tmp_path):
    with patch("main.load_surprisal_model") as load:
        with pytest.raises(ValueError, match="run record"):
            run_stuff(infile, "", parameters=PARAMS, rejection_file="review.csv")
    load.assert_not_called()


def test_uncommitted_changes_list_modified_and_untracked_files_but_not_ignored_ones(tmp_path, monkeypatch):
    """An uncommitted new module changes a run as much as an edited one."""
    import run_record

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    (tmp_path / ".gitignore").write_text("output/\n")
    (tmp_path / "old.py").write_text("x = 1\n")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    (tmp_path / "old.py").write_text("x = 2\n")
    (tmp_path / "new.py").write_text("y = 1\n")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "out.csv").write_text("a\n")
    monkeypatch.setattr(run_record, "GENERATOR_DIR", str(tmp_path))

    assert run_record.generator_commit()["uncommitted_changes"] == ["new.py", "old.py"]
