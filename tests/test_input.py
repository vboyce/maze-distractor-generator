import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from input import read_input


def write(tmp_path, text):
    f = tmp_path / "in.csv"
    f.write_text(text)
    return str(f)


def test_groups_sentences_by_item_number(tmp_path):
    sents = read_input(write(tmp_path, "type,item_num,sentence\na,1,The cat sat.\nb,1,The dog ran.\nc,2,A bird.\n"))
    assert sorted(sents) == ["1", "2"]
    assert len(sents["1"].sentences) == 2


def test_accepts_column_name_aliases(tmp_path):
    sents = read_input(write(tmp_path, "Condition,ID,Sentence\na,7,The cat sat.\n"))
    sentence = sents["7"].sentences[0]
    assert sentence.tag == "a"
    assert sentence.words == ["The", "cat", "sat."]


def test_labels_column_is_used_when_present(tmp_path):
    sents = read_input(write(tmp_path, "type,item_num,sentence,labels\na,1,The cat sat.,pre cat verb\n"))
    assert sents["1"].sentences[0].labels == ["pre", "cat", "verb"]


def test_labels_of_the_wrong_length_raise(tmp_path):
    with pytest.raises(ValueError):
        read_input(write(tmp_path, "type,item_num,sentence,labels\na,1,The cat sat.,pre cat\n"))


def test_missing_sentence_column_raises(tmp_path):
    with pytest.raises(ValueError, match="sentence"):
        read_input(write(tmp_path, "type,item_num,text\na,1,The cat sat.\n"))
