import csv
from sentence_set import Sentence, Sentence_Set


def read_input(filename):
    """Read an input CSV with a header row into sentence sets.

    Columns, matched case-insensitively, with the alternatives in brackets:
    type [tag, condition, group], copied to the output unchanged;
    item_num [itemnum, id, item, item_id], sentences with the same item
    number get matched distractors; sentence; and optionally labels, one per
    word (without them, words are labelled by position: "0", "1", ...).

    Blank lines are skipped. A row with too few cells, an empty item number
    or sentence, or the wrong number of labels is an error naming its line.

    Returns: a dict of item number -> Sentence_Set, in order of first appearance.
    """
    all_sentences = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        header = [col.strip() for col in next(reader)]  # Read header row and strip whitespace

        # Create mapping from column names (case-insensitive, normalized) to indices
        header_map = {}
        for idx, col_name in enumerate(header):
            normalized = col_name.lower().replace('_', '').replace('-', '')
            header_map[normalized] = idx

        # Find expected columns (try variations)
        def find_col(variants):
            for variant in variants:
                normalized = variant.strip().lower().replace('_', '').replace('-', '')
                if normalized in header_map:
                    return header_map[normalized]
            return None

        tag_idx = find_col(['type', 'tag', 'condition', 'group'])
        id_idx = find_col(['item_num', 'itemnum', 'id', 'item', 'item_id'])
        sentence_idx = find_col(['sentence', "sentences"])
        labels_idx = find_col(['labels', 'label'])

        # Validate required columns
        if tag_idx is None:
            raise ValueError(f"Could not find 'type'/'tag' column in header: {header}")
        if id_idx is None:
            raise ValueError(f"Could not find 'item_num'/'id' column in header: {header}")
        if sentence_idx is None:
            raise ValueError(f"Could not find 'sentence' column in header: {header}")
        required_cells = max(tag_idx, id_idx, sentence_idx) + 1

        for row in reader:
            row = [cell.strip() for cell in row]
            if not any(row):
                continue  # a blank line
            line = f"{filename}, line {reader.line_num}"
            if len(row) < required_cells:
                raise ValueError(f"{line}: {len(row)} cells, but the header needs at least {required_cells}: {row}")
            tag = row[tag_idx]
            id = row[id_idx]
            word_sentence = row[sentence_idx]
            if not id:
                raise ValueError(f"{line}: the item number is empty")
            if not word_sentence:
                raise ValueError(f"{line}: the sentence is empty")
            words = word_sentence.split()

            if labels_idx is not None and labels_idx < len(row) and row[labels_idx]:
                labels = row[labels_idx].split()
                if len(labels) != len(words):
                    raise ValueError(
                        f"{line} (item {id}): {len(labels)} labels for {len(words)} words in sentence '{word_sentence}'"
                    )
            else:
                # Default labels are word positions, as strings so they match
                # labels read back from CSV files (e.g. a review file).
                labels = [str(i) for i in range(len(words))]

            if id not in all_sentences.keys():
                all_sentences[id] = Sentence_Set(id)
            all_sentences[id].add(Sentence(words, labels, id, tag))
    return all_sentences
