#!/usr/bin/env python3
"""Report the word-length distribution of a word list and drop words of
MAX_LENGTH or more characters, writing the result to a new file.

This was one step in building curated_word_list.txt (the default include list);
most curation (removing offensive and sensitive words) was done by hand.

Usage:
    python scripts/curate_wordlist.py IN_FILE OUT_FILE [--max-length 15]
"""
import argparse
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("infile", help="word list, one word per line")
    parser.add_argument("outfile", help="where to write the filtered list")
    parser.add_argument("--max-length", type=int, default=15,
                        help="drop words with this many characters or more (default: 15)")
    args = parser.parse_args()

    with open(args.infile) as f:
        words = [line.strip() for line in f if line.strip()]

    length_counts = Counter(len(w) for w in words)
    print(f"{'Length':>6}  {'Count':>6}")
    print("-" * 16)
    for length in sorted(length_counts):
        print(f"{length:>6}  {length_counts[length]:>6}")
    print(f"\n Total: {len(words)}")

    kept = [w for w in words if len(w) < args.max_length]
    print(f"\nRemoved {len(words) - len(kept)} words with {args.max_length}+ characters")
    with open(args.outfile, "w") as f:
        for w in kept:
            f.write(w + "\n")
    print(f"Wrote {len(kept)} words to {args.outfile}")


if __name__ == "__main__":
    main()
