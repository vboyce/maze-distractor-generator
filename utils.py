def _alnum_bounds(word):
    """Return (first, last) indices of the alphanumeric core of word, or None if
    word has no letters or digits."""
    alnum_positions = [i for i, ch in enumerate(word) if ch.isalnum()]
    if not alnum_positions:
        return None
    return alnum_positions[0], alnum_positions[-1]


def strip_punct(word):
    '''take a word, return word with start and end punctuation removed'''
    bounds = _alnum_bounds(word)
    if bounds is None:
        return ""
    first, last = bounds
    return word[first:last + 1]


def copy_punct(word, distractor):
    """Takes the start and end punctuation of word as well as the capitalization pattern
    and return distractor with that punctuation and capitalization.
    A word with no letters or digits (e.g. "--") is kept whole as a prefix."""
    bounds = _alnum_bounds(word)
    if bounds is None:
        return word + distractor.lower()
    first, last = bounds
    prefix = word[:first]
    suffix = word[last + 1:]
    core = word[first:last + 1]
    if len(core) > 1 and core.isupper():
        distractor = distractor.upper()  # all capitalized
    elif len(core) > 1 and core[0].isupper():
        distractor = distractor[0:1].upper() + distractor[1:]  # first letter capitalized
    else:
        distractor = distractor.lower()  # all lowercase
    return prefix + distractor + suffix
