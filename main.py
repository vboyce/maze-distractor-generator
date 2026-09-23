import logging
import csv
import os
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal, load_surprisal_model
from input import read_input
from output import (save_delim, save_json, save_distractor_summary, save_longform,
                    updated_longform_rows, write_longform_rows, assign_distractors_from_longform)
from utils import strip_punct

SURPRISAL_LOG_HEADER = ["record_type", "sentence_set_id", "label", "prefix", "word", "surprisal_target", "actual_surprisal", "met_threshold"]


def read_rejection_file(filepath):
    """Read a longform review CSV and return locked and rejected position info.

    A label is rejected for an item if any of its rows is marked in the
    "rejected" column (a label can appear in several sentences of an item,
    which share one distractor). All other labels are locked to their current
    distractor.

    Returns:
        locked_by_item: dict mapping item_num (str) -> {label (str): distractor (str, with punctuation)}
        rejected_items: set of item_num strings that have at least one rejected label.
    """
    with open(filepath, newline="") as f:
        rows = list(csv.DictReader(f))

    rejected_labels = {(row["item_num"], row["label"]) for row in rows if row.get("rejected", "").strip()}
    rejected_items = {item_num for item_num, _ in rejected_labels}

    locked_by_item = {}
    for row in rows:
        key = (row["item_num"], row["label"])
        if key in rejected_labels:
            continue
        locked_by_item.setdefault(row["item_num"], {})[row["label"]] = row["distractor"]
    return locked_by_item, rejected_items


def seed_repeats_from_locked(repeats, locked_by_item, rejected_items):
    """Count the locked distractors of items that won't be regenerated.

    Items being regenerated count their own locked distractors in
    Sentence_Set.do_distractors, so they are skipped here to avoid counting
    those words twice.
    """
    for item_num, locked in locked_by_item.items():
        if item_num in rejected_items:
            continue
        for dist_word in locked.values():
            repeats.increment(strip_punct(dist_word).lower())


def run_stuff(infile, outfile, logfile=None, summaryfile=None, parameters="params.txt", outformat="delim", module_name="stimuli",
              model_override=None, backend_override=None,
              rejection_file=None, num_options=1, longform_outfile=None):
    """Generate distractors for every sentence in infile and write them to outfile.
    
    Args:
        infile: Input CSV file with sentences
        outfile: Output file path
        logfile: Optional CSV file to log distractor candidates with their surprisals
        parameters: A params file path, a dict of parameters, or None for all defaults
        outformat: Output format ('delim' or 'json')
        module_name: Name for JSON export (if outformat='json')
        summaryfile: Optional CSV file counting how often each distractor was used
        model_override: If set, use this model name instead of the one in params
        backend_override: If set, use this backend instead of the one in params
        rejection_file: Optional longform review CSV with a "rejected" column. Only
            items with a rejected label are regenerated, and only their rejected
            labels change. The sentence-level output then covers every sentence,
            built from the updated review rows.
        num_options: Number of candidates per position to list in the longform output
        longform_outfile: Optional path for longform (one row per word position) output
    """
    if outformat not in ["delim", "json"]:
        raise ValueError("outfile format not understood: " + outformat)
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params["dictionary_loc"]),
                         params["dictionary_class"])
    d = dict_class(params)
    model_name = model_override or params["model"]
    backend_name = backend_override or params["backend"]
    backend = load_surprisal_model(model_name, backend=backend_name)
    threshold_func = getattr(importlib.import_module(params["threshold_loc"]),
                             params["threshold_name"])
    repeats = Repeatcounter(params["max_repeat"])

    # Rejection mode: pre-seed repeat counter and identify which sentences to process
    locked_by_item = {}
    rejected_items = None
    if rejection_file:
        locked_by_item, rejected_items = read_rejection_file(rejection_file)
        seed_repeats_from_locked(repeats, locked_by_item, rejected_items)

    log_writer = None
    log_handle = None
    if logfile:
        parent = os.path.dirname(logfile)
        if parent:
            os.makedirs(parent, exist_ok=True)
        log_handle = open(logfile, "w", newline="")
        log_writer = csv.writer(log_handle)
        log_writer.writerow(SURPRISAL_LOG_HEADER)

    try:
        for ss in sents.values():
            if rejected_items is not None and ss.id not in rejected_items:
                continue  # skip sentences with no rejections
            locked = locked_by_item.get(ss.id, {})
            logging.info("Processing sentence_set_id %s", ss.id)
            ss.do_surprisals(backend, log_writer=log_writer)
            ss.make_labels()
            ss.do_distractors(backend, d, threshold_func, params, repeats,
                              locked=locked, num_options=num_options, log_writer=log_writer)
    finally:
        if log_handle:
            log_handle.close()

    if rejection_file:
        # In rejection mode, the review file is the record of every distractor:
        # merge the regenerated ones into it, then build the sentence-level output
        # (all sentences) from it.
        processed_sents = {id: ss for id, ss in sents.items() if id in rejected_items}
        fieldnames, rows = updated_longform_rows(rejection_file, processed_sents)
        if longform_outfile:
            write_longform_rows(longform_outfile, fieldnames, rows)
        assign_distractors_from_longform(sents, rows)
        if outfile:
            if outformat == "json":
                save_json(outfile, sents, module_name)
            else:
                save_delim(outfile, sents)
    else:
        if outformat == "json":
            save_json(outfile, sents, module_name)
        else:
            save_delim(outfile, sents)
        if longform_outfile:
            save_longform(longform_outfile, sents)

    if summaryfile:
        save_distractor_summary(summaryfile, sents)
