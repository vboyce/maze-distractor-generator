import logging
import ast

DEFAULTS = {
    "min_delta": 10,
    "min_abs": 25,
    "num_to_test": 100,
    "dictionary_loc": "wordfreq_distractor",
    "dictionary_class": "wordfreq_dict",
    "threshold_loc": "wordfreq_distractor",
    "threshold_name": "get_thresholds",
    "backend": "transformers",
    "max_repeat": 0,
    "language": "en",
    "fallback_ranking": "gap",
    "prepend_start_token": True,
    "revision": None,
    "candidate_order": "random",
    "seed": 0,
    "avoid_item_words": True,
    "check_shown_case": True,
}

# Keys whose value must be one of a fixed set.
ALLOWED_VALUES = {
    "fallback_ranking": ("gap", "min_surprisal"),
    "prepend_start_token": (True, False),
    "candidate_order": ("closest", "random"),
    "avoid_item_words": (True, False),
    "check_shown_case": (True, False),
}

# Numeric keys: (allowed types, smallest allowed value or None). bool is
# excluded separately, since True is an int in Python.
NUMBER_RULES = {
    "num_to_test": ((int,), 1),
    "max_repeat": ((int,), 0),
    "min_delta": ((int, float), 0),
    "min_abs": ((int, float), 0),
    "seed": ((int,), None),
}

# Heuristics that use English word lists. Each defaults to on when language is
# "en" and off otherwise; turning one on for another language is an error.
ENGLISH_ONLY_KEYS = ("avoid_names_and_abbreviations", "check_often_capitalized", "break_noun_phrases")

# Keys with no default that must be given. There is no default model, so that
# a run can't silently use a model nobody chose.
REQUIRED_KEYS = {"model"}

# Keys with no default that the dictionary class reads if present.
OPTIONAL_KEYS = {"include_words", "exclude_words"}


def _read_params_file(file):
    """Parse a params file into a dict: one `key: value` per line, split at
    the first colon (so a value may contain colons), the value a Python
    literal. Blank lines and lines starting with # are skipped."""
    params = {}
    with open(file, "r") as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{file}, line {number}: expected 'key: value', got {line!r}")
            key, value = (part.strip() for part in line.split(":", 1))
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError) as error:
                raise ValueError(f"{file}, line {number}: the value of {key} is not a Python literal "
                                 f"(strings need quotes): {value!r}") from error
    return params


def set_params(source=None, overrides=None):
    """Build a complete parameter dictionary from a file path, a dict, or defaults.

    Args:
        source: One of:
            - str: path to a colon-delimited params file
            - dict: parameter values
            - None: no values (defaults only; the required keys must then come
              from overrides)
        overrides: Optional dict of values that win over source (e.g. from
            command-line flags). Entries whose value is None are ignored.

    Returns:
        dict with all parameters filled in (user values override defaults).

    Raises:
        ValueError: if a required key is missing, a key is unknown, or a
            value is of the wrong type or out of range.
    """
    if source is None:
        user_params = {}
    elif isinstance(source, dict):
        user_params = dict(source)
    else:
        user_params = _read_params_file(source)
    if overrides:
        user_params.update({key: value for key, value in overrides.items() if value is not None})

    missing = REQUIRED_KEYS - set(user_params)
    if missing:
        raise ValueError(f"Missing required parameter(s) {sorted(missing)}.")

    params = {}
    for key, default in DEFAULTS.items():
        if key in user_params:
            params[key] = user_params.pop(key)
        else:
            logging.info("Using default %s = %s", key, default)
            params[key] = default

    for key in ENGLISH_ONLY_KEYS:
        english = params["language"] == "en"
        if key in user_params:
            params[key] = user_params.pop(key)
        else:
            logging.info("Using default %s = %s", key, english)
            params[key] = english
        if params[key] not in (True, False):
            raise ValueError(f"{key} must be True or False, not {params[key]!r}")
        if params[key] and not english:
            raise ValueError(f"{key} uses English word lists, so it can't be True for language {params['language']!r}")

    unknown = set(user_params) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)}. Known parameters: "
            f"{sorted(set(DEFAULTS) | set(ENGLISH_ONLY_KEYS) | REQUIRED_KEYS | OPTIONAL_KEYS)}"
        )
    params.update(user_params)
    for key, allowed in ALLOWED_VALUES.items():
        if params[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, not {params[key]!r}")
    for key, (types, minimum) in NUMBER_RULES.items():
        check_number(key, params[key], types, minimum)
    return params


def check_number(key, value, types, minimum):
    if isinstance(value, bool) or not isinstance(value, types):
        kind = "a whole number" if types == (int,) else "a number"
        raise ValueError(f"{key} must be {kind}, not {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be at least {minimum}, not {value!r}")
