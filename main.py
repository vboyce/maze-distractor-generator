import logging
import time
import csv
import os
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal, load_surprisal_model
from input import read_input
from output import save_delim, save_json

SURPRISAL_LOG_HEADER = ["record_type", "sentence_set_id", "label", "prefix", "word", "surprisal_target", "actual_surprisal", "met_threshold"]

def run_stuff(infile, outfile, logfile=None, parameters="params.txt", outformat="delim", module_name="stimuli"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat
    
    Args:
        infile: Input CSV file with sentences
        outfile: Output file path
        logfile: Optional CSV file to log distractor candidates with their surprisals
        parameters: Parameter file path
        outformat: Output format ('delim' or 'json')
        module_name: Name for JSON export (if outformat='json')
    """
    if outformat not in ["delim", "json"]:
        raise ValueError("outfile format not understood: " + outformat)
    params = set_params(parameters)
    sents = read_input(infile)
    dict_class = getattr(importlib.import_module(params.get("dictionary_loc", "wordfreq_distractor")),
                         params.get("dictionary_class", "wordfreq_English_dict"))
    d = dict_class(params)
    model_name = params.get("model", "gpt2")
    model, tokenizer = load_surprisal_model(model_name)
    threshold_func = getattr(importlib.import_module(params.get("threshold_loc", "wordfreq_distractor")),
                             params.get("threshold_name", "get_thresholds"))
    repeats = Repeatcounter(params.get("max_repeat", 0))

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
            logging.info("Processing sentence_set_id %s", ss.id)
            ss.do_surprisals(model, tokenizer, log_writer=log_writer)
            ss.make_labels()
            ss.do_distractors(model, tokenizer, d, threshold_func, params, repeats, log_writer=log_writer)
    finally:
        if log_handle:
            log_handle.close()
    if outformat == "json":
        save_json(outfile, sents, module_name)
    else:
        save_delim(outfile, sents)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()

    run_stuff("input/test_in.csv", "output/test_output.csv", logfile="output/verbose.csv", parameters="params.txt", outformat="delim", module_name="STIM")
    logging.info("run_stuff completed in %.2fs", time.perf_counter() - t0)