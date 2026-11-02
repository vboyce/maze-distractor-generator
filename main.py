import logging
import csv
import os
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal, load_surprisal_model
from input import read_input
from output import (save_delim, save_json, save_distractor_summary, save_longform,
                    read_review_rows, updated_longform_rows, write_longform_rows, assign_distractors_from_longform,
                    report_rows, save_report)
from utils import strip_punct
from sentence_set import is_no_distractor
from run_record import default_record_path, make_run_record, save_run_record

SURPRISAL_LOG_HEADER = ["record_type", "sentence_set_id", "label", "prefix", "word", "surprisal_target",
                        "actual_surprisal", "met_threshold", "surprisal_exact"]


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
    _, rows = read_review_rows(filepath)
    rejected_labels = {(row["item_num"], row["label"]) for row in rows if row["rejected"].strip()}
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
            if not is_no_distractor(dist_word):
                repeats.increment(strip_punct(dist_word).lower())


def log_shortfalls(rows):
    """Log each report row whose distractor fell short of its target (info),
    and how many did (one warning)."""
    short = [row for row in rows if not row["met"]]
    for row in short:
        logging.info("Item %s, label %s: %s fell short of its target %.1f (surprisal %s)",
                     row["item_num"], row["label"], row["distractor"], row["threshold"], row["surprisal"])
    if short:
        logging.warning("%d of %d positions fell short of their target; see --report for which",
                        len(short), len(rows))


def run_stuff(infile, outfile, logfile=None, summaryfile=None, parameters="params.txt", outformat="delim", module_name="stimuli",
              model_override=None, backend_override=None,
              rejection_file=None, num_options=1, longform_outfile=None, report_outfile=None,
              run_record_outfile=None):
    """Generate distractors for every sentence in infile and write them to outfile.
    
    Args:
        infile: Input CSV file with sentences
        outfile: Output file path
        logfile: Optional CSV file to log distractor candidates with their surprisals
        parameters: A params file path, a dict of parameters, or None (then
            model_override must name the model)
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
        report_outfile: Optional CSV with each position's frequencies, surprisal
            target and distractor surprisal (see output.report_rows); in
            rejection mode, only the regenerated positions
        run_record_outfile: Where to write the run record (see run_record.py);
            default <outfile>.run.json, or <longform_outfile>.run.json in
            rejection mode with no outfile
    """
    if outformat not in ["delim", "json"]:
        raise ValueError("outfile format not understood: " + outformat)
    run_record_outfile = run_record_outfile or default_record_path(outfile, longform_outfile)
    params = set_params(parameters, overrides={"model": model_override, "backend": backend_override})
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params["dictionary_loc"]),
                         params["dictionary_class"])
    d = dict_class(params)
    if params["check_often_capitalized"] and not hasattr(d, "often_capitalized"):
        raise ValueError(
            f"check_often_capitalized is on, but dictionary_class {params['dictionary_class']} has no "
            "often_capitalized attribute (the set of words that must also meet the target capitalized). "
            "Give the class one, or set check_often_capitalized: False.")
    if params["break_noun_phrases"] and not hasattr(d, "noun_phrase_breakers"):
        raise ValueError(
            f"break_noun_phrases is on, but dictionary_class {params['dictionary_class']} has no "
            "noun_phrase_breakers attribute (the set of words that may follow a determiner). "
            "Give the class one, or set break_noun_phrases: False.")
    backend = load_surprisal_model(params["model"], backend=params["backend"],
                                   prepend_start_token=params["prepend_start_token"],
                                   revision=params["revision"])
    run_record = make_run_record(params, backend, d, infile, rejection_file)
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

    processed_sents = sents if rejected_items is None else {
        id: ss for id, ss in sents.items() if id in rejected_items}
    position_report = report_rows(processed_sents, params["language"])
    log_shortfalls(position_report)
    if report_outfile:
        save_report(report_outfile, position_report)

    if rejection_file:
        # In rejection mode, the review file is the record of every distractor:
        # merge the regenerated ones into it, then build the sentence-level output
        # (all sentences) from it.
        review_fieldnames, review_rows = updated_longform_rows(rejection_file, processed_sents)
        if longform_outfile:
            write_longform_rows(longform_outfile, review_fieldnames, review_rows)
        assign_distractors_from_longform(sents, review_rows)
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
    save_run_record(run_record_outfile, run_record)
