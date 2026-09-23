import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import read_rejection_file, seed_repeats_from_locked
from input import read_input
from limit_repeats import Repeatcounter


def write(tmp_path, name, text):
    f = tmp_path / name
    f.write_text(text)
    return str(f)


class TestLabelTypes:
    def test_rejection_file_accepts_string_labels(self, tmp_path):
        f = write(tmp_path, "review.csv", """\
type,item_num,label,prefix,real_word,distractor,options,rejected
sub_rel,3,pre_2,The,cat,dog,,
sub_rel,3,who,The cat,who,ran,,yes
""")
        locked, rejected = read_rejection_file(f)
        assert locked == {"3": {"pre_2": "dog"}}
        assert rejected == {"3"}

    def test_default_input_labels_match_rejection_file_labels(self, tmp_path):
        """Labels from read_input (no labels column) must be the same type as the
        labels read back from a review file, so locked positions are found."""
        infile = write(tmp_path, "in.csv", "type,item_num,sentence\nx,1,The cat sat.\n")
        review = write(tmp_path, "review.csv", """\
type,item_num,label,prefix,real_word,distractor,options,rejected
x,1,1,The,cat,dog,,
x,1,2,The cat,sat.,ran.,,yes
""")
        sents = read_input(infile)
        locked, _ = read_rejection_file(review)
        input_labels = sents["1"].sentences[0].labels
        assert set(locked["1"]) <= set(input_labels)


class TestSharedLabels:
    def test_rejecting_one_row_of_a_shared_label_rejects_the_label(self, tmp_path):
        """Two sentences in an item share label 'verb'; rejecting it in one row
        must regenerate it (not keep the other row's distractor locked)."""
        f = write(tmp_path, "review.csv", """\
type,item_num,label,prefix,real_word,distractor,options,rejected
a,1,verb,The dog,barked,fish,,yes
b,1,verb,The big dog,barked,fish,,
b,1,adj,The,big,soft,,
""")
        locked, rejected = read_rejection_file(f)
        assert "verb" not in locked["1"]
        assert locked["1"]["adj"] == "soft"


class TestSeedRepeats:
    def test_only_items_not_being_regenerated_are_seeded(self):
        """Locked words of items being regenerated are counted by do_distractors,
        so seeding them here too would count them twice."""
        repeats = Repeatcounter(0)
        locked_by_item = {"1": {"1": "Dog"}, "2": {"1": "fish,"}}
        seed_repeats_from_locked(repeats, locked_by_item, rejected_items={"2"})
        assert repeats.distractors == {"dog": 1}
