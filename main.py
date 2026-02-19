import logging
import time
import importlib
from set_params import set_params
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal, load_surprisal_model
from input import read_input
from output import  save_delim
import os.path

def run_stuff(infile, outfile, parameters="params.txt", outformat="delim"):
    """Takes an input file, and an output file location
    Does the whole distractor thing (according to specified parameters)
    Writes in outformat"""
    if outformat not in ["delim"]:
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
    repeats=Repeatcounter(params.get("max_repeat", 0))
    print(repeats.max)
    print(repeats.limit)
    for ss in sents.values():
        ss.do_surprisals(model, tokenizer)
        ss.make_labels()
        ss.do_distractors(model, tokenizer, d, threshold_func, params, repeats)

    # display the results
    for ss in sents.values():
        for sentence in ss.sentences:
            print(sentence.word_sentence)
            print(sentence.distractor_sentence)
            print(sentence.label_sentence)
            print("--------------------------------")
    #if outformat == "delim":
    #    save_delim(outfile, sents)
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.perf_counter()
    run_stuff("test_input.txt", "test_output.txt", "params.txt", "delim")
    logging.info("run_stuff completed in %.2fs", time.perf_counter() - t0)