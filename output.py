import csv
import os

def save_delim(outfile, all_sentences):
    '''Saves results to a file in semicolon delimited format
    basically same as the original input with another column for distractor sentence
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    Returns: none
    will write a semicolon delimited file with
    column 1 = "tag"/condition copied over from item_to_info (from input file)
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

import json


def save_json(outfile, all_sentences, name="stimuli"):
    '''Saves results as a JavaScript module (for use with jspsych)
    Arguments:
    outfile = location of a file to write to
    all_sentences: dictionary of sentence_set objects
    name: variable name for the exported stimuli list (optional)
    Returns: none
    Writes a .js file with "export const stimuli = [...]" where each item has:
    * item_type (tag/condition)
    * id (item number)
    * sentence (original sentence)
    * distractor (distractor sentence)
    * labels (label string)
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
                "sentence": sentence.word_sentence,
                "distractor": sentence.distractor_sentence,
                "labels": sentence.label_sentence,
            })
    js_content = f"export const {name} = " + json.dumps(items, indent=2) + ";\n"
    with open(outfile, "w") as f:
        f.write(js_content)

