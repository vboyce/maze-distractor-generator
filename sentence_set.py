import logging
import csv
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal



def no_duplicates(my_list):
    """True if list has no duplicates, else false"""
    return len(my_list) == len(set(my_list))


class Sentence:
    """a sentence to get distractors for
    has a list of words (in the sentence)
    a list of labels for matching with other sentences
    an id = item number
    and tag, which we do nothing with, but store anyhow
    """

    def __init__(self, words, labels, id, tag):
        if no_duplicates(labels):
            self.words = words  # list of words in sentence
            self.word_sentence = " ".join(self.words)  # sentence itself
            self.labels = labels  # list of labels in sentence
            self.label_sentence = " ".join([str(lab) for lab in self.labels])  # labels as sentence
            self.id = id  # item number
            self.tag = tag  # group/type
            self.distractors = ["x-x-x"]  # the first word has no context, so it gets a placeholder distractor
            self.distractor_sentence = ""
            self.probs = {}  # using a dictionary so we can start at 1 and not 0
            self.surprisal = {}
        else:
            raise ValueError("duplicate labels on sentence " + " ".join(words) + " with labels " + " ".join(labels))

    def do_surprisal(self, backend, log_writer=None, sentence_set_id=None):
        """Get surprisals of words in sentence"""
        set_id = sentence_set_id if sentence_set_id is not None else self.id
        for i in range(1, len(self.labels)):  #we don't care about the surprisal of the first word
            lab = self.labels[i]
            prefix = " ".join(self.words[:i])
            self.surprisal[lab] = get_surprisal(backend=backend, prefix=prefix, word=self.words[i])
            if log_writer:
                log_writer.writerow(["real_word", set_id, lab, prefix, self.words[i], "", self.surprisal[lab], ""])
            


class Label:
    """A set of words etc, associated with a label, within a sentence set"""

    def __init__(self, id, lab):
        self.id = id  # item number
        self.lab = lab
        self.words = []
        self.prefixes = []
        self.surprisals = []
        self.surprisal_targets = []

    def add_sentence(self, word, prefix, surprisal):
        """Given a position that belongs in the label, add it's attributes to our lists"""
        self.words.append(word)
        self.prefixes.append(prefix)
        self.surprisals.append(surprisal)

    def choose_top_n_distractors(self, n, backend, dict, threshold_func, params, banned, sentence_set_id, log_writer=None):
        """Return up to n distractor candidates, best first.

        A candidate is "good" if its surprisal meets the target
        (max(min_abs, real word's surprisal + min_delta)) after every prefix
        with this label. Good candidates come first, in the order found; then
        the rest, by their lowest surprisal, highest first. The search stops
        once n good candidates are found. Candidates on `banned` and the real
        words themselves are skipped.

        Sets self.distractor to the first result ("x-x-x" if there are none).
        """
        for surprisal in self.surprisals:
            self.surprisal_targets.append(max(params["min_abs"], surprisal + params["min_delta"]))
        min_length, max_length, min_freq, max_freq = threshold_func(self.words, params)
        distractor_opts = dict.get_potential_distractors(min_length, max_length, min_freq, max_freq, params)
        avoid = [strip_punct(word).lower() for word in self.words]

        good = []
        fallback = []  # (min_surp, word)

        for dist in distractor_opts:
            if len(good) >= n:
                break
            if dist in banned or dist in avoid:
                continue
            dist_good = True
            min_surp = float("inf")
            for i in range(len(self.words)):
                dist_surp = get_surprisal(backend=backend, prefix=self.prefixes[i], word=dist)
                if log_writer:
                    log_writer.writerow([
                        "distractor_candidate", sentence_set_id, self.lab,
                        self.prefixes[i], dist, self.surprisal_targets[i],
                        dist_surp, dist_surp >= self.surprisal_targets[i]
                    ])
                if dist_surp < self.surprisal_targets[i]:
                    dist_good = False
                    min_surp = min(min_surp, dist_surp)
            if dist_good:
                good.append(dist)
            else:
                fallback.append((min_surp, dist))

        fallback.sort(reverse=True, key=lambda x: x[0])
        result = (good + [w for _, w in fallback])[:n]
        self.distractor = result[0] if result else "x-x-x"
        return result

    def choose_distractor(self, backend, dict, threshold_func, params, banned, sentence_set_id, log_writer=None):
        """Choose one distractor: the first candidate that meets the surprisal
        target in every sentence with this label, or else the candidate whose
        lowest surprisal is highest. Candidates on `banned` (used too often, or
        already used in this sentence set) and the real words are skipped.
        Returns "x-x-x" if there are no candidates."""
        self.choose_top_n_distractors(1, backend, dict, threshold_func, params, banned, sentence_set_id, log_writer)
        return self.distractor


class Sentence_Set:
    """A set of sentence objects, with the same id"""

    def __init__(self, id):
        self.id = id
        self.sentences = []
        self.label_ids = set()
        self.first_labels = set()
        self.labels = {}  # dictionary of label:label object
        self.label_options = {}  # label -> list of candidate words (populated by do_distractors)

    def add(self, sentence):
        """Adds a sentence item to the sentence_set"""
        if sentence.id == self.id:
            self.sentences.append(sentence)
            first_label = sentence.labels[0]
            self.first_labels = self.first_labels.union(set([first_label]))
            self.label_ids = self.label_ids.union(sentence.labels[1:])
            if self.first_labels & self.label_ids != set():
                raise ValueError("Labels of first words cannot match labels of later words in the same set in item " + str(self.id))
        else:
            raise ValueError("ID doesn't match for item " + str(sentence.id) + " and item " + str(self.id))

    def do_surprisals(self, backend, log_writer=None):
        """Gets surprisals for the real words"""
        for sentence in self.sentences:
            sentence.do_surprisal(backend, log_writer=log_writer, sentence_set_id=self.id)

    def make_labels(self):
        """Regroups the stuff in the sentence items into by-label groups"""
        for lab in self.label_ids: #init label objects
            self.labels[lab] = Label(self.id, lab)
        for sentence in self.sentences: #dump stuff into the label objects
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                self.labels[lab].add_sentence(sentence.words[i], " ".join(sentence.words[:i]), sentence.surprisal[lab])

    def do_distractors(self, backend, d, threshold_func, params, repeats, locked=None, num_options=1, log_writer=None):
        """Get distractors using specified stuff.

        locked: optional dict mapping label -> distractor word (with punctuation) for positions to keep as-is.
        num_options: number of candidate options to generate per unlocked position (default 1 = original behavior).
        """
        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        self.label_options = {}
        for label in self.labels.values():
            if locked and label.lab in locked:
                bare = strip_punct(locked[label.lab]).lower()
                label.distractor = bare
                banned.append(bare)
                repeats.increment(bare)
                self.label_options[label.lab] = [bare]
            elif num_options > 1:
                opts = label.choose_top_n_distractors(num_options, backend, d, threshold_func, params, banned, self.id, log_writer)
                if opts:
                    banned.append(opts[0])
                    repeats.increment(opts[0])
                self.label_options[label.lab] = opts
            else:
                dist = label.choose_distractor(backend, d, threshold_func, params, banned, self.id, log_writer)
                banned.append(dist)
                repeats.increment(dist)
                self.label_options[label.lab] = [dist]
        for sentence in self.sentences: #give the sentences the distractors
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                distractor = copy_punct(sentence.words[i], self.labels[lab].distractor)
                sentence.distractors.append(distractor)
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format