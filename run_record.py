"""The run record: a JSON file, written beside the output of every run, with
what is needed to reproduce it (code, model, package versions, parameters,
and checksums of the files read)."""
import hashlib
import json
import logging
import os
import platform
import subprocess
from importlib import metadata

GENERATOR_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDED_PACKAGES = ("torch", "transformers", "wordfreq")


def default_record_path(outfile, longform_outfile):
    """<outfile>.run.json, or beside the longform output in rejection mode
    with no outfile."""
    if outfile:
        return f"{outfile}.run.json"
    if longform_outfile:
        return f"{longform_outfile}.run.json"
    raise ValueError("No output to write the run record beside: give an outfile, a longform outfile, "
                     "or run_record_outfile (--run-record)")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def file_entry(path):
    return {"path": path, "sha256": sha256(path)}


def generator_commit():
    """The generator's git commit, and the files that differ from it
    (modified, or untracked and not ignored), sorted; [] if none. Both None,
    with a warning, if the generator is not a git checkout."""
    head = subprocess.run(["git", "-C", GENERATOR_DIR, "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0:
        logging.warning("The generator is not a git checkout, so the run record has no commit: %s",
                        head.stderr.strip())
        return {"commit": None, "uncommitted_changes": None}
    # -z: one NUL-separated "XY path" entry per file, paths unquoted.
    status = subprocess.run(["git", "-C", GENERATOR_DIR, "status", "--porcelain", "-z", "--untracked-files=all",
                             "--no-renames"], capture_output=True, text=True, check=True)
    changed = sorted(entry[3:] for entry in status.stdout.split("\0") if entry)
    return {"commit": head.stdout.strip(), "uncommitted_changes": changed}


def word_list_entries(dictionary, dictionary_class):
    """Checksums of the word lists the dictionary read; None, with a warning,
    for a dictionary class that does not list them (word_list_files)."""
    if not hasattr(dictionary, "word_list_files"):
        logging.warning("dictionary_class %s has no word_list_files, so the run record has no word lists",
                        dictionary_class)
        return None
    return [file_entry(path) for path in dictionary.word_list_files]


def make_run_record(params, backend, dictionary, infile, rejection_file=None):
    """The run record, as JSON text. Raises ValueError for a value JSON
    cannot hold (inf or nan), before the run rather than after it."""
    record = {
        "generator": generator_commit(),
        "model": {"name": params["model"], "backend": params["backend"], "revision": params["revision"],
                  "commit": backend.model_commit},
        "versions": {"python": platform.python_version(),
                     **{package: metadata.version(package) for package in RECORDED_PACKAGES}},
        "parameters": params,
        "files": {"input": file_entry(infile),
                  "word_lists": word_list_entries(dictionary, params["dictionary_class"]),
                  "rejection_file": file_entry(rejection_file) if rejection_file else None},
    }
    try:
        return json.dumps(record, indent=2, allow_nan=False)
    except ValueError as error:
        raise ValueError(f"The run record can't hold a value of inf or nan (check the parameters): {error}") from error


def save_run_record(path, record_text):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(record_text + "\n")
