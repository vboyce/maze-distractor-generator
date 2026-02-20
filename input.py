import logging
import csv
from sentence_set import Sentence, Sentence_Set


def read_input(filename):
    '''Reads an input CSV file with a header row
    Arguments:
    filename = a comma-delimited CSV file with a header row and the following columns:
    first column (type) = any info that should stay associated with the sentence such as condition etc
    this will be copied to eventual output unchanged
    second column (item_num) = item number, all sentences with the same item number will get corresponding distractors based on labels (or sentence position) 
    Third column (sentence) = sentence
    Fourth column (labels) = labels; if it exists, must be same number of words as sentence. if it doesn't exist,
    will be given 1:n labels ()
    Returns:
    item_to_info = a dictionary of item numbers as keys and a pair of lists (conditions, sentences) as value
    sentences =  a list of sentences grouped by item number (ie will get matching distractors)'''
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
        
        for row in reader:
            # Strip whitespace from each cell
            row = [cell.strip() for cell in row]
            
            tag = row[tag_idx] if tag_idx < len(row) else ""
            id = row[id_idx] if id_idx < len(row) else ""
            word_sentence = row[sentence_idx] if sentence_idx < len(row) else ""
            words = word_sentence.split()
            
            if labels_idx is not None and labels_idx < len(row) and row[labels_idx]:
                label_sentence = row[labels_idx]
                labels = label_sentence.split()
                if len(labels) != len(words):
                    if len(labels) == 0:
                        labels = list(range(0, len(words)))
                    else:
                        logging.error("Labels are wrong length for sentence %s", word_sentence)
                        raise ValueError
            else:
                labels = list(range(0, len(words)))
            
            if id not in all_sentences.keys():
                all_sentences[id] = Sentence_Set(id)
            all_sentences[id].add(Sentence(words, labels, id, tag))
    return all_sentences
