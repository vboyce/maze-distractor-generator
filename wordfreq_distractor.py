import wordfreq
import re
import math
import random
import logging

import utils
from distractor import distractor_dict, distractor

def check_language(language):
    """Raise if wordfreq has no frequency data for this language code."""
    if language not in wordfreq.available_languages():
        raise ValueError(
            f"wordfreq has no data for language '{language}'. "
            f"Available: {sorted(wordfreq.available_languages())}"
        )


def read_word_list(path):
    """Read a word list, one word per line."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


class wordfreq_dict(distractor_dict):
    """Candidate distractors, with frequencies from wordfreq for params["language"]
    (a wordfreq language code, default "en").

    A word is a candidate if it is in the include list (params["include_words"]),
    in wordfreq's vocabulary for the language, not in the exclude list
    (params["exclude_words"]), and made only of lowercase letters (so proper
    nouns, abbreviations and words with digits or punctuation are dropped; for
    English, only a-z).

    English defaults to curated_word_list.txt; other languages must name their
    own include list.
    """

    def __init__(self, params={}):
        self.language = params.get("language", "en")
        check_language(self.language)
        default_include = "curated_word_list.txt" if self.language == "en" else None
        include = params.get("include_words", default_include)
        if include is None:
            raise ValueError(
                f"language '{self.language}' needs its own word list: set include_words "
                "to a file of candidate distractor words, one per line"
            )
        exclude = params.get("exclude_words", "exclude.txt")

        freqs = wordfreq.get_frequency_dict(self.language)
        exclusions = set(read_word_list(exclude)) if exclude is not None else set()
        words = (set(read_word_list(include)) & set(freqs)) - exclusions
        self.words = []
        for word in sorted(words):
            if self.allowed(word):
                # we canonically calculate frequency as log occurrences / 1 billion words
                self.words.append(distractor(word, math.log(freqs[word] * 10 ** 9)))

    def allowed(self, word):
        """Whether a word may be used as a distractor."""
        if self.language == "en":
            return re.match("^[a-z]+$", word) is not None
        return word.isalpha() and word.islower()

    def in_dict(self, test_word):
        """Test to see if word is in dictionary"""
        for word in self.words:
            if word.text == test_word:
                return word
        return False

    def get_words(self, length_low, length_high, freq_low, freq_high):
        """Returns a list of words within specified ranges"""
        matches = []
        for word in self.words:
            if freq_low <= word.freq <= freq_high and length_low <= word.len <= length_high:
                matches.append(word.text)
        return matches

    def get_potential_distractors(self, min_length, max_length, min_freq, max_freq, params):
        """returns list of n words, if possible from between threshold values
        if not tries things nearby -- higher frequency and then lower"""
        distractor_opts = self.get_words(min_length, max_length, min_freq, max_freq)
        random.shuffle(distractor_opts)
        n=params['num_to_test']
        if len(distractor_opts) >= n:
            return distractor_opts[:n]
        else:
            logging.info("Having to widen distractor option search")
            still_need = n - len(distractor_opts)
            i = 1
            while i < 10:
                new = []
                lower = self.get_words(min_length, max_length, min_freq - i, min_freq - i + 1)
                higher = self.get_words(min_length, max_length, max_freq + i - 1, max_freq + i)
                new.extend(lower)
                new.extend(higher)
                random.shuffle(new)
                if len(new) >= still_need:
                    distractor_opts.extend(new)
                    return distractor_opts[:n]
                distractor_opts.extend(new)
                i += 1
        logging.warning("Could not find enough distractors: found %d of %d", len(distractor_opts), n)
        return distractor_opts


class wordfreq_English_dict(wordfreq_dict):
    """English-only version of wordfreq_dict, kept for parameter files that name it."""

    def __init__(self, params={}):
        if params.get("language", "en") != "en":
            raise ValueError(
                "wordfreq_English_dict only supports language 'en'; "
                "use dictionary_class: \"wordfreq_dict\" for other languages"
            )
        super().__init__(params)


def get_frequency(word, language="en"):
    """Frequency of word in language, on the same scale as the dictionary
    (log occurrences per billion words)."""
    return wordfreq.zipf_frequency(word, language) * math.log(10)  # rescale to fit


def get_thresholds(words, params=None):
    """Given the real words at a position, return (min_length, max_length,
    min_freq, max_freq) bounds for candidate distractors. Frequencies are for
    params["language"] (default "en")."""
    language = (params or {}).get("language", "en")
    check_language(language)
    lengths = []
    freqs = []
    for word in words:
        stripped = utils.strip_punct(word)
        lengths.append(len(stripped))
        freqs.append(get_frequency(stripped, language))
    min_length = min(min(lengths)-1, 12)
    max_length = max(max(lengths)+1, 4)
    min_freq = min(min(freqs)-1, 11)
    max_freq = max(max(freqs)+1, 3)
    return min_length, max_length, min_freq, max_freq
