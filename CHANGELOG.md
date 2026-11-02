# Changelog

## 2026-11 (branch `fixes-2026-10`)

Distractors generated before this change and after it differ, even with the
same parameter file. To reproduce older distractors, use the commit they were
made with; for example, the kid-maze-passages experiment 1 stimuli were made
with commit `c1efc39` or earlier. From now on, each run writes a run record
(below) with the commit it used.

### Changes to the distractors chosen

These are new defaults. Each has a parameter, so the old behavior can be
had back, except where noted.

- **Start token** (`prepend_start_token: True`): a causal model reads its
  start token before the sentence, as it did at the start of each document
  in training. This lowers the surprisal of words early in a sentence, so it
  changes the targets as well as the candidates' scores. From Amani
  Maina-Kilaas's generator.
- **Fallback ranking** (`fallback_ranking: "gap"`): when no candidate meets
  the target, the one whose worst shortfall is smallest is chosen. Before,
  fallbacks were ranked on their lowest surprisal, and only over the
  sentences they fell short in, not all of them as the README said; that bug
  is fixed (`"min_surprisal"` now ranks on every sentence). The bug was
  reported by Guangyuan Jiang.
- **Item words avoided** (`avoid_item_words: True`): no word of any sentence
  in the item is used as a distractor, including parts of words ("cat's"
  rules out "cat"). From Amani Maina-Kilaas's generator.
- **Candidate order** (`candidate_order: "random"`, `seed: 0`): candidates
  are still tried in random order, but the order is now seeded, so a run can
  be repeated exactly. `"closest"` (closest in length, then frequency, first)
  is available. From Amani Maina-Kilaas's generator.
- **Capitalization** (`check_shown_case: True`): a distractor shown
  capitalized, to match its real word, must meet the target as shown as well
  as in lower case.
- **English-only heuristics**, on by default for English only, with word
  lists from Amani Maina-Kilaas's generator (see `word_lists/README.md`):
  - `avoid_names_and_abbreviations`: words readers know only as a first name
    or an abbreviation are never used;
  - `check_often_capitalized`: a word often read as a name, place or title
    ("grace", "oxford") must also meet the target capitalized;
  - `break_noun_phrases`: after a determiner ("the", "a", "his", ...), only
    words that can't continue a noun phrase are tried.
- **No parameter, always on:**
  - A position with no candidate to try gets `NO-DISTRACTOR`, with a
    warning. Before, it got `x-x-x`, the same as the first word.
  - Words that will be skipped (already used, or in the item) are dropped
    before the candidates are capped at `num_to_test`, not after, so they no
    longer crowd out usable words.
  - Labels are processed in the order they first appear. Before, their order
    could change from one run to the next.

### Changes that may break existing setups

- **`model` is required.** There is no default model; before, it fell back
  to `gpt2`.
- **Word lists moved** to `word_lists/` (`word_lists/curated_word_list.txt`,
  `word_lists/exclude.txt`). Parameter files that name the old paths must be
  updated; the defaults are found wherever you run from.
- **New errors:** an English-only heuristic set to `True` for another
  language, and `prepend_start_token: True` (the default) with the `litellm`
  backend, which must set it to `False`.
- **A custom `dictionary_class`** must have an `often_capitalized` attribute
  when `check_often_capitalized` is on, and a `noun_phrase_breakers`
  attribute when `break_noun_phrases` is on.
- **Models are loaded as float32**, and a sentence longer than the model can
  read is an error. From Amani Maina-Kilaas's generator.

### Fixed: wrong output, or errors that didn't say what was wrong

- A position kept as `NO-DISTRACTOR` through review is kept as the marker.
  Before, it was lowercased, shown in the real word's case
  ("No-distractor") and counted as a used distractor.
- A model whose config names no causal or masked LM architecture is an error
  that says to set `backend`. Before, it was loaded as causal with a
  warning, so a model without an LM head would get a random one.
- Parameter files are split at the first colon, so a value may contain one
  (`model: "ollama/llama3:8b"`). Before, `num_to_test: 100: 5` was read as
  100. A line without a colon, or a value that is not a Python literal, is
  an error naming the line; blank lines are skipped (they used to crash).
- `num_to_test` must be a whole number of at least 1, `max_repeat` a whole
  number of at least 0, `min_delta` and `min_abs` numbers of at least 0, and
  `seed` a whole number.
- The `litellm` backend raises on a word token with no log-probability.
  Before, it skipped the token, understating surprisal.
- An input row that is short a cell, or has an empty item number or
  sentence, is an error naming the line; blank rows are skipped. Before, a
  missing cell was read as empty, and a blank row crashed with an
  unexplained `IndexError`.
- A review file (`--rejection-file`) without the `item_num`, `label`,
  `prefix`, `distractor` and `rejected` columns is an error. Before, a file
  with no `rejected` column was read as "nothing rejected".
- The masked-LM backend checks that a sentence fits the model, as the causal
  backend does; before, a too-long sentence failed with an indexing error.
- `check_distractors.py` stops, before any API call, at a row with a
  different number of distractors than words. Before, a missing distractor
  was recorded as judged ungrammatical.
- `benchmark.py` stops at a model that fails. Before, it logged the error
  and went on to the next model.

### New

- **Faster scoring** on causal models: candidates are scored in batches.
  On `input/test_ns_9_short.csv` with `EleutherAI/pythia-160m` on a CPU, a
  run took 46 s against 667 s one candidate at a time (14.5 times faster,
  one run each), with the same distractors. The design is from Amani
  Maina-Kilaas's generator.
- **`revision`** pins the Hugging Face model version.
- **Run record**, written every run beside the output (`OUTPUT.run.json`):
  the generator commit, model and resolved model commit, package versions,
  every parameter, and checksums of the input and word lists.
- **`--report FILE`**: each position's frequencies, surprisal target,
  distractor surprisal and whether it met the target. Every run now ends
  with one warning counting the positions that fell short.
- **`--log FILE`** has a `surprisal_exact` column: with batch scoring, some
  logged surprisals are lower bounds.
- `check_distractors.py` reads the Anthropic API key from a `.env` file if it
  isn't set in the environment, and stops before any API call without one.
