import csv
import os
from collections import Counter

import json
import wordfreq
from utils import strip_punct, copy_punct
from sentence_set import NO_DISTRACTOR

LONGFORM_FIELDNAMES = ["type", "item_num", "label", "prefix", "real_word", "distractor", "options", "rejected"]
REPORT_FIELDNAMES = ["type", "item_num", "label", "position", "word", "distractor", "word_zipf", "distractor_zipf",
                     "threshold", "surprisal", "met"]


def save_delim(outfile, all_sentences):
    '''Saves results to a comma-separated CSV file with a header row
    basically same as the original input with another column for distractor sentence
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none
    will write a CSV with columns type, item_num, sentence, distractors, labels:
    column 1 = "tag"/condition copied over from the input file
    column 2 = item number
    column 3 = good sentence
    column 4 = string of distractor words in order.
    column 5 = string of labels in order. '''
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)  
    with open(outfile, 'w+', newline="") as f:
        writer=csv.writer(f,delimiter=",")
        writer.writerow(["type", "item_num", "sentence", "distractors", "labels"])
        for sentence_set in all_sentences.values():
            for sentence in sentence_set.sentences:
                writer.writerow([sentence.tag,sentence.id,sentence.word_sentence,sentence.distractor_sentence,sentence.label_sentence])




def save_json(outfile, all_sentences, name="stimuli"):
    '''Saves results as a JavaScript module (for use with jspsych)
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    name: variable name for the exported stimuli list (optional)
    Returns: none
    Writes a .js file with "export const <name> = [...]" where each item has:
    * item_type (tag/condition)
    * id (item number)
    * sent (original sentence)
    * distractor (distractor sentence)
    * labels (label string)
    The sent/distractor names match what the jspsych-maze demos read.
    '''
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    items = []
    for sentence_set in all_sentences.values():
        for sentence in sentence_set.sentences:
            items.append({
                "item_type": sentence.tag,
                "id": sentence.id,
                "sent": sentence.word_sentence,
                "distractor": sentence.distractor_sentence,
                "labels": sentence.label_sentence,
            })
    js_content = f"export const {name} = " + json.dumps(items, indent=2) + ";\n"
    with open(outfile, "w") as f:
        f.write(js_content)


def save_longform(outfile, all_sentences):
    """Save results in longform: one row per non-first-word position across all sentences.

    Columns: type, item_num, label, prefix, real_word, distractor, options, rejected

    'distractor' is the assigned word (with punctuation applied).
    'options' is a comma-separated list of alternative candidates beyond the first
    (populated when num_options > 1 was used; empty otherwise).
    'rejected' is always empty on initial output — the reviewer fills this in.
    """
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONGFORM_FIELDNAMES)
        writer.writeheader()
        for ss in all_sentences.values():
            for sentence in ss.sentences:
                for i in range(1, len(sentence.labels)):
                    lab = sentence.labels[i]
                    real_word = sentence.words[i]
                    distractor = sentence.distractors[i]
                    opts = ss.label_options.get(lab, [])
                    # opts[0] is already reflected in distractor; opts[1:] are alternatives
                    alt_strings = [copy_punct(real_word, o) for o in opts[1:]]
                    writer.writerow({
                        "type": sentence.tag,
                        "item_num": sentence.id,
                        "label": lab,
                        "prefix": " ".join(sentence.words[:i]),
                        "real_word": real_word,
                        "distractor": distractor,
                        "options": ", ".join(alt_strings),
                        "rejected": "",
                    })


# A review file without these can't say which positions to regenerate or
# where their new distractors go; without "rejected" it would silently
# regenerate nothing.
REVIEW_REQUIRED_COLUMNS = ["item_num", "label", "prefix", "distractor", "rejected"]


def read_review_rows(review_path):
    """Read a longform review CSV. Returns (fieldnames, rows); raises if a
    required column is missing."""
    with open(review_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    missing = [column for column in REVIEW_REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"{review_path} is missing the column(s) {missing}; "
                         f"is it a review file written by --longform?")
    return fieldnames, rows


def updated_longform_rows(original_longform_path, all_sentences):
    """Merge newly generated distractors into the rows of a longform review file.

    A label is regenerated for an item if any of its rows is marked rejected, so
    every row with that (item, label) gets its new distractor (with its own
    sentence's punctuation), its options, and a cleared 'rejected' flag. Other
    rows are unchanged. The 'options' column is added if missing.

    all_sentences should contain only the sentence sets that were regenerated.

    Returns (fieldnames, rows).
    """
    orig_fieldnames, orig_rows = read_review_rows(original_longform_path)
    rejected_keys = {(row["item_num"], row["label"]) for row in orig_rows if row["rejected"].strip()}

    # (item_num, label, prefix) identifies one word position in one sentence.
    new_data = {}
    for ss in all_sentences.values():
        for sentence in ss.sentences:
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                real_word = sentence.words[i]
                prefix = " ".join(sentence.words[:i])
                opts = ss.label_options.get(lab, [])
                alt_strings = [copy_punct(real_word, o) for o in opts[1:]]
                new_data[(str(sentence.id), str(lab), prefix)] = (sentence.distractors[i], ", ".join(alt_strings))

    out_fieldnames = list(orig_fieldnames)
    if "options" not in out_fieldnames:
        out_fieldnames.insert(out_fieldnames.index("rejected"), "options")

    rows = []
    for row in orig_rows:
        row = dict(row)
        if (row["item_num"], row["label"]) in rejected_keys:
            key = (row["item_num"], row["label"], row["prefix"])
            if key not in new_data:
                raise ValueError(f"No regenerated distractor for item {key[0]}, label {key[1]}, prefix '{key[2]}'")
            row["distractor"], row["options"] = new_data[key]
            row["rejected"] = ""
        rows.append(row)
    return out_fieldnames, rows


def write_longform_rows(outfile, fieldnames, rows):
    """Write longform rows (as returned by updated_longform_rows) to a CSV."""
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_updated_longform(outfile, original_longform_path, all_sentences):
    """Update a longform review file with newly generated distractors for rejected
    labels (see updated_longform_rows) and write it to outfile."""
    fieldnames, rows = updated_longform_rows(original_longform_path, all_sentences)
    write_longform_rows(outfile, fieldnames, rows)


def assign_distractors_from_longform(all_sentences, rows):
    """Set every sentence's distractors from longform rows (e.g. a reviewed file),
    so the sentence-level output reflects the review. Raises if a word position
    has no row."""
    by_position = {(row["item_num"], row["label"], row["prefix"]): row["distractor"] for row in rows}
    for ss in all_sentences.values():
        for sentence in ss.sentences:
            distractors = ["x-x-x"]
            for i in range(1, len(sentence.labels)):
                key = (str(sentence.id), str(sentence.labels[i]), " ".join(sentence.words[:i]))
                if key not in by_position:
                    raise ValueError(
                        f"The review file has no row for item {key[0]}, label {key[1]}, prefix '{key[2]}'. "
                        "Was it made from this input file?"
                    )
                distractors.append(by_position[key])
            sentence.distractors = distractors
            sentence.distractor_sentence = " ".join(distractors)


def save_distractor_summary(outfile, all_sentences):
    """Saves a CSV summarizing distractor usage: each distinct distractor and its count.

    Args:
        outfile: path to the output CSV file
        all_sentences: dictionary of sentence_set objects
    """
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    counts = Counter()
    for sentence_set in all_sentences.values():
        for sentence in sentence_set.sentences:
            for word in sentence.distractors:
                counts[strip_punct(word).lower()] += 1
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["distractor", "count"])
        for word in sorted(counts):
            writer.writerow([word, counts[word]])


def report_rows(all_sentences, language):
    """One row per word after the first, for every label whose distractor was
    chosen in this run (not one kept from an earlier run in rejection mode):
    the word and its distractor as shown, their Zipf frequencies in language,
    the surprisal target, the distractor's surprisal there (the lowest of the
    forms scored), and whether it met the target. A NO_DISTRACTOR position
    has no distractor frequency or surprisal, and did not meet its target.

    all_sentences: dictionary of sentence_set objects, each with its labels
        made and its distractors chosen.
    """
    rows = []
    for sentence_set in all_sentences.values():
        # A label's targets and surprisals are in the order of the sentences
        # it appears in (see Sentence_Set.make_labels).
        times_seen = Counter()
        for sentence in sentence_set.sentences:
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                label = sentence_set.labels[lab]
                k = times_seen[lab]
                times_seen[lab] += 1
                if label.distractor_surprisals is None:
                    continue
                surprisal = label.distractor_surprisals[k]
                threshold = label.surprisal_targets[k]
                rows.append({
                    "type": sentence.tag,
                    "item_num": sentence.id,
                    "label": lab,
                    "position": i,
                    "word": sentence.words[i],
                    "distractor": sentence.distractors[i],
                    "word_zipf": wordfreq.zipf_frequency(strip_punct(sentence.words[i]), language),
                    "distractor_zipf": ("" if label.distractor == NO_DISTRACTOR
                                        else wordfreq.zipf_frequency(label.distractor, language)),
                    "threshold": threshold,
                    "surprisal": "" if surprisal is None else surprisal,
                    "met": surprisal is not None and surprisal >= threshold,
                })
    return rows


def save_report(outfile, rows):
    """Write report_rows to a CSV file."""
    parent = os.path.dirname(outfile)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
