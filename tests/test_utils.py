import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import strip_punct, copy_punct


class TestStripPunct:
    def test_removes_leading_and_trailing_punctuation(self):
        assert strip_punct('"Hello,"') == "Hello"

    def test_keeps_internal_punctuation(self):
        assert strip_punct("long-bearded.") == "long-bearded"

    def test_empty_string(self):
        assert strip_punct("") == ""

    def test_all_punctuation(self):
        assert strip_punct("--") == ""


class TestCopyPunct:
    def test_copies_punctuation_and_initial_capital(self):
        assert copy_punct('"Hello,"', "dog") == '"Dog,"'

    def test_copies_all_caps(self):
        assert copy_punct("NASA.", "dog") == "DOG."

    def test_lowercase_word_gives_lowercase_distractor(self):
        assert copy_punct("cat", "Dog") == "dog"

    def test_empty_word(self):
        assert copy_punct("", "dog") == "dog"

    def test_all_punctuation_word_is_kept_as_a_prefix(self):
        assert copy_punct("--", "dog") == "--dog"
