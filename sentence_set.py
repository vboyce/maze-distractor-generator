import logging
import csv
import re
from utils import copy_punct, strip_punct
from limit_repeats import Repeatcounter
from get_surprisal import get_surprisal

FALLBACK_RANKINGS = ("gap", "min_surprisal")

# Shown in place of a distractor when a position had no candidate to try.
NO_DISTRACTOR = "NO-DISTRACTOR"


# Words that are always determiners, so a noun phrase is open after them;
# "that", "this" and "her" are as often something else. From Amani
# Maina-Kilaas's maze-distractor-generator (generation.py), GPL-3.
DETERMINERS = frozenset(["the", "a", "an", "my", "your", "his", "its", "our", "their", "every"])


def ends_in_determiner(prefix):
    """Whether the last word of prefix is one of DETERMINERS, with no
    punctuation after it: in "It was his." the noun phrase is over."""
    last = prefix.split()[-1]
    word = strip_punct(last)
    return last.endswith(word) and word.lower() in DETERMINERS


def item_words(words):
    """The lowercase letter runs of words, so "Cat's" gives "cat" and "s",
    and "well-known" gives "well" and "known"."""
    return {part for word in words for part in re.findall(r"[^\W\d_]+", word.lower())}


def forms_to_score(target_word, candidate, check_shown_case, often_capitalized):
    """The forms of a (lowercase) candidate that must each meet the target
    at target_word's position: always the lowercase form; with
    check_shown_case, also the form it is shown in (its case copied from
    target_word, see copy_punct); and if it is often_capitalized, also
    capitalized, since a reader may take "grace" for the name Grace.

    The model can find a capitalized word unsurprising as a name where a
    reader who knows the word would not, and the reverse. Ported, in outline,
    from Amani Maina-Kilaas's maze-distractor-generator (generation.py
    _variant_keys), GPL-3.
    """
    forms = [candidate]
    if check_shown_case:
        forms.append(strip_punct(copy_punct(target_word, candidate)))
    if often_capitalized:
        forms.append(candidate.capitalize())
    return list(dict.fromkeys(forms))

# Backends with a score_candidates method score this many candidates at
# once, then twice as many, and so on, until enough good ones are found.
FIRST_CHUNK = 16


def score_candidates(backend, prefix, candidates, exact_below):
    """(surprisal, exact) for each candidate after prefix.

    Backends with a fast path (the causal transformers backend) define
    score_candidates, which may return a lower bound (exact=False) for a
    score at or above exact_below. Every other backend is scored one
    candidate at a time with get_surprisal, exactly.
    """
    if hasattr(backend, "score_candidates"):
        return backend.score_candidates(prefix, candidates, exact_below)
    return [(get_surprisal(backend=backend, prefix=prefix, word=word), True) for word in candidates]


def candidate_chunks(candidates, batched):
    """Consecutive chunks of candidates: FIRST_CHUNK, then doubling, if
    batched; otherwise one at a time."""
    start, size = 0, FIRST_CHUNK if batched else 1
    while start < len(candidates):
        yield candidates[start:start + size]
        start += size
        if batched:
            size *= 2


def fallback_rank(surprisals, targets, fallback_ranking):
    """Ranking value of a candidate that misses some target; lower is better.

    "gap": its worst shortfall, max over contexts of (target - surprisal).
    "min_surprisal": its lowest surprisal in any context, negated (so the
    highest lowest surprisal ranks first).
    """
    if fallback_ranking == "gap":
        return max(target - surp for surp, target in zip(surprisals, targets))
    if fallback_ranking == "min_surprisal":
        return -min(surprisals)
    raise ValueError(f"Unknown fallback_ranking {fallback_ranking!r}; use one of {FALLBACK_RANKINGS}")


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
                log_writer.writerow(["real_word", set_id, lab, prefix, self.words[i], "", self.surprisal[lab], "", True])
            


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

    def choose_top_n_distractors(self, n, backend, dictionary, threshold_func, params, banned, sentence_set_id, log_writer=None):
        """Return up to n distractor candidates, best first.

        A candidate is "good" if its surprisal meets the target
        (max(min_abs, real word's surprisal + min_delta)) after every prefix
        with this label, in every form scored there (see forms_to_score:
        lowercase, and with params["check_shown_case"] and
        params["check_often_capitalized"], capitalized). Good candidates come first, in the order found; then
        the rest, ranked by params["fallback_ranking"] (see fallback_rank),
        ties in the order found. The search stops
        once n good candidates are found. Candidates on `banned` and the real
        words themselves are skipped. With params["break_noun_phrases"], if
        any prefix ends in one of DETERMINERS, only the dictionary's
        noun_phrase_breakers are candidates.

        Sets self.distractor to the first result, or NO_DISTRACTOR (with a
        warning) if there was no candidate to try.
        """
        for surprisal in self.surprisals:
            self.surprisal_targets.append(max(params["min_abs"], surprisal + params["min_delta"]))
        min_length, max_length, min_freq, max_freq = threshold_func(self.words, params)
        avoid = [strip_punct(word).lower() for word in self.words]
        # After a determiner, a noun, adjective, participle or adverb could
        # go on to make a sentence however surprising it is ("The alluded...").
        # `only` is passed just then, so a dictionary class written before
        # break_noun_phrases still works with it off.
        restriction = {}
        if params["break_noun_phrases"] and any(ends_in_determiner(prefix) for prefix in self.prefixes):
            restriction = {"only": dictionary.noun_phrase_breakers}
        distractor_opts = dictionary.get_potential_distractors(min_length, max_length, min_freq, max_freq, params,
                                                               target_words=self.words,
                                                               order_key=f"{self.id}\0{self.lab}",
                                                               avoid=list(banned) + avoid, **restriction)

        candidates = [dist for dist in distractor_opts if dist not in banned and dist not in avoid]
        # A score that meets its target may be a lower bound, which is all the
        # gap ranking needs; min_surprisal ranks fallbacks on exact scores.
        if params["fallback_ranking"] == "min_surprisal":
            exact_below = [float("inf")] * len(self.words)
        else:
            exact_below = self.surprisal_targets

        listed = dictionary.often_capitalized if params["check_often_capitalized"] else set()
        positions = range(len(self.words))

        good = []
        fallback = []  # (rank, word); lower rank is better

        for chunk in candidate_chunks(candidates, batched=hasattr(backend, "score_candidates")):
            if len(good) >= n:
                break
            # forms[k][i]: the forms of chunk[k] that must meet the target after self.prefixes[i]
            forms = [[forms_to_score(self.words[i], dist, params["check_shown_case"], dist in listed)
                      for i in positions] for dist in chunk]
            # scores[i][form]: (surprisal, exact) of form after self.prefixes[i]
            scores = []
            for i in positions:
                to_score = list(dict.fromkeys(form for dist_forms in forms for form in dist_forms[i]))
                scores.append(dict(zip(to_score, score_candidates(backend, self.prefixes[i], to_score, exact_below[i]))))
            for k, dist in enumerate(chunk):
                if len(good) >= n:
                    break
                # (surprisal, target) of each form at each position
                results = [(scores[i][form][0], self.surprisal_targets[i]) for i in positions for form in forms[k][i]]
                if log_writer:
                    for i in positions:
                        for form in forms[k][i]:
                            dist_surp, exact = scores[i][form]
                            log_writer.writerow([
                                "distractor_candidate", sentence_set_id, self.lab,
                                self.prefixes[i], form, self.surprisal_targets[i],
                                dist_surp, dist_surp >= self.surprisal_targets[i], exact
                            ])
                if all(surp >= target for surp, target in results):
                    good.append(dist)
                else:
                    rank = fallback_rank([surp for surp, _ in results], [target for _, target in results],
                                         params["fallback_ranking"])
                    fallback.append((rank, dist))

        fallback.sort(key=lambda x: x[0])  # stable, so ties stay in the order found
        result = (good + [w for _, w in fallback])[:n]
        if result:
            self.distractor = result[0]
        else:
            logging.warning("Item %s, label %s: no candidate distractors to try; using %s",
                            self.id, self.lab, NO_DISTRACTOR)
            self.distractor = NO_DISTRACTOR
        return result

    def choose_distractor(self, backend, dictionary, threshold_func, params, banned, sentence_set_id, log_writer=None):
        """Choose one distractor: the first candidate that meets the surprisal
        target in every sentence with this label, or else the best fallback
        (see choose_top_n_distractors). Candidates on `banned` (used too often, or
        already used in this sentence set) and the real words are skipped.
        Returns NO_DISTRACTOR if there are no candidates."""
        self.choose_top_n_distractors(1, backend, dictionary, threshold_func, params, banned, sentence_set_id, log_writer)
        return self.distractor


class Sentence_Set:
    """A set of sentence objects, with the same id"""

    def __init__(self, id):
        self.id = id
        self.sentences = []
        self.label_ids = {}  # used as an ordered set: labels in order of first appearance
        self.first_labels = set()
        self.labels = {}  # dictionary of label:label object
        self.label_options = {}  # label -> list of candidate words (populated by do_distractors)

    def add(self, sentence):
        """Adds a sentence item to the sentence_set"""
        if sentence.id == self.id:
            self.sentences.append(sentence)
            first_label = sentence.labels[0]
            self.first_labels = self.first_labels.union(set([first_label]))
            self.label_ids.update(dict.fromkeys(sentence.labels[1:]))
            if self.first_labels & set(self.label_ids) != set():
                raise ValueError("Labels of first words cannot match labels of later words in the same set in item " + str(self.id))
        else:
            raise ValueError("ID doesn't match for item " + str(sentence.id) + " and item " + str(self.id))

    def do_surprisals(self, backend, log_writer=None):
        """Gets surprisals for the real words"""
        for sentence in self.sentences:
            sentence.do_surprisal(backend, log_writer=log_writer, sentence_set_id=self.id)

    def make_labels(self):
        """Regroups the stuff in the sentence items into by-label groups"""
        for lab in self.label_ids: #init label objects, in order of first appearance
            self.labels[lab] = Label(self.id, lab)
        for sentence in self.sentences: #dump stuff into the label objects
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                self.labels[lab].add_sentence(sentence.words[i], " ".join(sentence.words[:i]), sentence.surprisal[lab])

    def do_distractors(self, backend, d, threshold_func, params, repeats, locked=None, num_options=1, log_writer=None):
        """Get distractors using specified stuff.

        locked: optional dict mapping label -> distractor word (with punctuation) for positions to keep as-is.
        num_options: number of candidate options to generate per unlocked position (default 1 = original behavior).

        With params["avoid_item_words"], no word of any sentence in the set
        (see item_words) is used as a distractor.
        """
        banned = repeats.banned[:] #don't allow duplicate distractors within the set
        if params["avoid_item_words"]:
            banned.extend(item_words(word for sentence in self.sentences for word in sentence.words))
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
                if dist != NO_DISTRACTOR:
                    banned.append(dist)
                    repeats.increment(dist)
                self.label_options[label.lab] = [dist]
        for sentence in self.sentences: #give the sentences the distractors
            for i in range(1, len(sentence.labels)):
                lab = sentence.labels[i]
                # we match distractors to their real words on punctuation
                dist = self.labels[lab].distractor
                distractor = copy_punct(sentence.words[i], dist, keep_case=(dist == NO_DISTRACTOR))
                sentence.distractors.append(distractor)
            sentence.distractor_sentence = " ".join(sentence.distractors) #and in sentence_format