import csv
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
}

# Keys whose value must be one of a fixed set.
ALLOWED_VALUES = {
    "fallback_ranking": ("gap", "min_surprisal"),
    "prepend_start_token": (True, False),
}

# Keys with no default that must be given. There is no default model, so that
# a run can't silently use a model nobody chose.
REQUIRED_KEYS = {"model"}

# Keys with no default that the dictionary class reads if present.
OPTIONAL_KEYS = {"include_words", "exclude_words"}


def _read_params_file(file):
    """Parse a colon-delimited params file into a dict."""
    params = {}
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=":", quotechar='"')
        for row in reader:
            if row and not row[0].startswith("#"):
                params[row[0].strip()] = ast.literal_eval(row[1].strip())
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
        ValueError: if a required key is missing or a key is unknown.
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

    unknown = set(user_params) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown parameter(s) {sorted(unknown)}. Known parameters: "
            f"{sorted(set(DEFAULTS) | REQUIRED_KEYS | OPTIONAL_KEYS)}"
        )
    params.update(user_params)
    for key, allowed in ALLOWED_VALUES.items():
        if params[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, not {params[key]!r}")
    return params
