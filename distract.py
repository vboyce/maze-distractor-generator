#!/usr/bin/env python3
"""Command-line interface: generate Maze distractors for a CSV of sentences.

Examples:
    python distract.py input/test_in.csv output/out.csv -p params.txt
    python distract.py input/test_in.csv output/stimuli.js --format json --module-name stimuli
    python distract.py input/test_in.csv output/out.csv --longform output/review.csv --num-options 3
    python distract.py input/test_in.csv --rejection-file output/review.csv --longform output/review_2.csv
"""
import argparse
import logging

from main import run_stuff


def build_parser():
    parser = argparse.ArgumentParser(description="Auto-generate Maze materials")
    parser.add_argument("input", help="input CSV (columns: type, item_num, sentence, optional labels)")
    parser.add_argument("output", nargs="?", default=None,
                        help="output file (sentence-level CSV, or a .js module with --format json); "
                             "optional in rejection mode")
    parser.add_argument("-p", "--parameters", default=None,
                        help="parameters file (default: built-in defaults)")
    parser.add_argument("--format", choices=["delim", "json"], default="delim",
                        help="delim: CSV; json: JavaScript module for jsPsych (default: delim)")
    parser.add_argument("--module-name", default="stimuli",
                        help="exported variable name for --format json (default: stimuli)")
    parser.add_argument("--model", default=None,
                        help="language model to use, overriding the parameters file")
    parser.add_argument("--backend", default=None,
                        help="surprisal backend (transformers, transformers_causal, transformers_masked, "
                             "litellm), overriding the parameters file")
    parser.add_argument("--longform", default=None, metavar="FILE",
                        help="also write longform output (one row per word position) to FILE, for review")
    parser.add_argument("--rejection-file", default=None, metavar="FILE",
                        help='longform CSV with a "rejected" column; regenerate only the marked positions')
    parser.add_argument("--num-options", type=int, default=1, metavar="N",
                        help="number of candidate distractors to list per position in the longform "
                             "output (default: 1)")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="write every candidate's surprisal to FILE (CSV)")
    parser.add_argument("--summary", default=None, metavar="FILE",
                        help="write a count of how often each distractor was used to FILE (CSV)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_stuff(
        args.input,
        args.output or "",
        logfile=args.log,
        summaryfile=args.summary,
        parameters=args.parameters,
        outformat=args.format,
        module_name=args.module_name,
        model_override=args.model,
        backend_override=args.backend,
        rejection_file=args.rejection_file,
        num_options=args.num_options,
        longform_outfile=args.longform,
    )


if __name__ == "__main__":
    main()
