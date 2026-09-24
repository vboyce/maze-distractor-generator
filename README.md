# maze-distractor-generator

Automatically generates distractor words for the **Maze task** (A-maze). For each word of each sentence, it picks a real English word that is a bad continuation of the sentence so far. It judges this with a language model (surprisal), while matching the correct word roughly on length and frequency.

This replaces the original A-maze code in [vboyce/Maze](https://github.com/vboyce/Maze) (`maze_automate`, now older). It works with current Hugging Face models (causal or masked) and API models, and adds a workflow for reviewing distractors and regenerating the bad ones.

- **Documentation:** https://vboyce.github.io/maze-docs
- **jsPsych plugin to run the task:** [jspsych-maze](https://github.com/vboyce/jspsych-maze)

## Install

Use Python 3.10 or later, in a project virtual environment:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
# optional: the litellm backend and check_distractors.py
.venv/bin/pip install -r requirements-optional.txt
```

On a machine without an NVIDIA GPU, install the CPU build of torch first (see the comment in `requirements.txt`). `requirements-lock.txt` has the exact versions of a known-good Linux + CUDA environment.

## Quick start

```sh
.venv/bin/python distract.py input/test_in.csv output/out.csv -p params.txt
```

This writes `output/out.csv`: the input sentences plus a `distractors` column. Run it from the repository root, because `params.txt` refers to the word lists by relative path. Models are downloaded from Hugging Face the first time they're used.

For jsPsych, write a JavaScript module instead:

```sh
.venv/bin/python distract.py input/test_in.csv output/stimuli.js --format json --module-name stimuli
```

Run `python distract.py --help` to see all options.

## Input

A CSV with a header row and these columns. Header names are matched case-insensitively, and the alternatives in brackets also work.

| Column | Contents |
|---|---|
| `type` [`tag`, `condition`, `group`] | Passed through to the output unchanged (condition, etc.). |
| `item_num` [`id`, `item`, `item_id`] | Item id. Sentences with the same id get matched distractors (see `labels`). |
| `sentence` | The sentence. Words are split on spaces; punctuation stays attached to its word. |
| `labels` (optional) | One label per word. Within an item, words with the same label get the same distractor. Without labels, words are matched by position. |

Example with labels:

```
type,item_num,sentence,labels
sub_rel,3,The cat who the dog scared hid in a box.,pre_1 pre_2 who art noun verb main_verb post_1 post_2 post_3
obj_rel,3,The dog who scared the cat sniffed around the couch.,pre_1 pre_2 who verb art noun main_verb post_1 post_2 post_3
```

Here the articles of the relative clauses get the same distractor, as do the verbs and the nouns, even though they are in different positions in the two sentences. The first word's label can't be reused later in the item, because the first word always gets the placeholder `x-x-x` (it has no context, so no distractor is meaningful).

## Output

- **`--format delim`** (default): a CSV with columns `type, item_num, sentence, distractors, labels`. `distractors` is a space-separated string with one distractor per word, starting with `x-x-x`.
- **`--format json`**: a JavaScript module, `export const <module-name> = [...]`. Each item has `item_type`, `id`, `sent`, `distractor` and `labels`. The `sent` / `distractor` keys match what the jspsych-maze demos read.
- **`--longform FILE`**: also write one row per word position (`type, item_num, label, prefix, real_word, distractor, options, rejected`), for review (see below).
- **`--summary FILE`**: how many times each distractor was used.
- **`--log FILE`**: every candidate that was tried, with its surprisal and whether it met the target. Useful for debugging or for choosing distractors by hand.

Distractors get the same leading/trailing punctuation and capitalization as the word they replace, so punctuation and case don't give the answer away.

## How distractors are chosen

For each label in an item:

1. **Target.** Compute the surprisal of the real word(s) given the sentence so far. The target for a distractor is `max(min_abs, real-word surprisal + min_delta)` bits, in every sentence where the label appears.
2. **Candidates.** Draw `num_to_test` random words from the word list whose length and frequency are close to the real word(s). If there aren't enough, widen the frequency band.
3. **Choice.** Take the first candidate that meets the target in every sentence. If none does, take the candidate whose lowest surprisal is highest. Candidates are skipped if they are the real word, are already used in the item, or have been used `max_repeat` times across the whole run.

Surprisal is calculated by tokenizing `prefix + " " + word`, comparing with the tokenization of `prefix` alone, and summing the surprisal of the extra tokens. This assumes it doesn't matter exactly how spaces are attached to tokens, which seems fair since only approximate surprisal matters. Real words are scored with their punctuation and capitalization; distractors are scored bare (lower case, no punctuation). That is usually conservative: end punctuation always raises surprisal.

## Parameters

Parameters come from a file passed with `-p` (colon-separated `key: value` lines; values are Python literals, so strings need quotes; `#` starts a comment). Keys you leave out get the defaults below. An unknown key is an error, so a typo or an out-of-date key can't be silently ignored. See `params.txt` for an example.

| Key | Default | Meaning |
|---|---|---|
| `min_delta` | `10` | A distractor must be at least this many bits more surprising than the real word... |
| `min_abs` | `25` | ...and at least this surprising in absolute terms. |
| `num_to_test` | `100` | Number of candidates drawn per position. |
| `model` | `"gpt2"` | Model name (Hugging Face id, or a liteLLM model string). |
| `backend` | `"transformers"` | See "Backends". |
| `max_repeat` | `0` | Maximum times any distractor is used in the whole run (`0` = no limit). `params.txt` uses `1`. |
| `language` | `"en"` | [wordfreq](https://github.com/rspeer/wordfreq) language code (e.g. `"fr"`, `"de"`, `"zh"`), used for word frequencies. See "Other languages". |
| `include_words` | `"curated_word_list.txt"` for English; required otherwise | Word list distractors are drawn from. |
| `exclude_words` | `"exclude.txt"` | Words never used as distractors. |
| `dictionary_loc`, `dictionary_class` | `"wordfreq_distractor"`, `"wordfreq_dict"` | Module and class that provide candidate words and frequencies. (`wordfreq_English_dict` still works, for English only.) |
| `threshold_loc`, `threshold_name` | `"wordfreq_distractor"`, `"get_thresholds"` | Module and function that turn the real words into length/frequency bounds. |

`--model` and `--backend` on the command line override the file.

## Backends

| `backend` | Models |
|---|---|
| `transformers` | Any Hugging Face causal or masked LM; the type is detected from the model config. |
| `transformers_causal` | Force causal (GPT-2, Pythia, Llama, …). |
| `transformers_masked` | Force masked (BERT, RoBERTa, …). A multi-token word is filled in left to right. |
| `litellm` | API models via [liteLLM](https://docs.litellm.ai/). The provider must return log-probabilities for the prompt (`echo=True`), which many chat APIs don't. Needs `requirements-optional.txt`. |

Models we have run (see `benchmark.py`): `gpt2`, `distilgpt2`, `EleutherAI/gpt-neo-125M`, `EleutherAI/pythia-160m`, `HuggingFaceTB/SmolLM-360M`, `Qwen/Qwen2-0.5B`, `distilbert-base-uncased`, `distilroberta-base`.

## Reviewing and regenerating distractors

Automatic distractors are sometimes plausible continuations, or unsuitable for your participants. To review them:

1. Generate with a longform review file, optionally listing extra candidates per position:
   ```sh
   .venv/bin/python distract.py input.csv out.csv -p params.txt --longform review.csv --num-options 3
   ```
2. Open `review.csv` and put anything (e.g. `x`) in the `rejected` column of each bad distractor.
3. Regenerate only the rejected positions:
   ```sh
   .venv/bin/python distract.py input.csv --rejection-file review.csv --longform review_2.csv -p params.txt
   ```
   Every position that wasn't rejected keeps its distractor. If a label appears in several sentences of an item, rejecting it in any one row regenerates it (in every row). Only items with a rejection are re-run.
4. Repeat 2–3 until you're happy. To also get the final sentence-level output (all sentences), give an output file in rejection mode; it is built from the updated review file:
   ```sh
   .venv/bin/python distract.py input.csv final.js --format json --rejection-file review_2.csv --longform review_3.csv -p params.txt
   ```
   If nothing is marked rejected, nothing is regenerated, and this just converts the review file.

Other options for quality control:
- **Experimental, in progress:** `check_distractors.py` asks an LLM whether each target and distractor is a grammatical continuation. It isn't reliable yet. In one test run (9 Natural Stories items, Claude Sonnet), 8% of the *real* words were judged ungrammatical, so don't use it to reject distractors automatically. `check-check-distractors.R` is exploratory code for looking at its output.
  ```sh
  .venv/bin/python check_distractors.py out.csv judgments.csv --model anthropic/claude-sonnet-4-20250514
  ```
  This makes one paid API call per word. It needs `requirements-optional.txt` and the provider's API key in the environment.
- Screen critical items (or all items) yourself by running through them in the task.
- Pilot on a few participants and regenerate distractors that several of them get wrong.

For many uses a few plausible distractors don't matter much. Filtering is worth it for high-stakes experiments, or for experiments with children.

## Word lists

`curated_word_list.txt` has 19.4K words of 1–14 characters, each occurring at least 2<sup>7</sup> times per billion words. They are filtered to "real" all-lower-case words, excluding offensive and sensitive (sexual, violent, religious) words. There are no guarantees, so review distractors for your own use. `scripts/curate_wordlist.py` is the length-filtering step used to build it. `exclude.txt` lists words that are never used. Word frequencies come from [wordfreq](https://github.com/rspeer/wordfreq).

### Other languages

Set `language` to a [wordfreq language code](https://github.com/rspeer/wordfreq#sources-and-supported-languages), point `include_words` at a word list for that language (one word per line; there is no default outside English), and choose a model trained on the language:

```
language: "fr"
include_words: "french_words.txt"
exclude_words: None
model: "<a French or multilingual Hugging Face model>"
```

Frequencies for both the candidates and the real words then come from that language. Candidates must be all-lowercase letters (accented letters are fine). An unknown language code, or a non-English language without `include_words`, is an error. Punctuation and capitalization are still copied from the real word as for English, which may not suit every language (e.g. German noun capitalization). See the docs page [A-maze in other languages](https://vboyce.github.io/maze-docs/non-english.html).

## Using it from Python

```python
from main import run_stuff
run_stuff("input.csv", "out.csv", parameters="params.txt", outformat="delim",
          longform_outfile="review.csv", logfile="log.csv")
```

`parameters` can also be a dict or `None` (all defaults).

## Tests

```sh
.venv/bin/python -m pytest
```

The tests use a fake surprisal backend, so they don't download models. `tests/test_end_to_end.py` runs the whole pipeline, including a review/regenerate round trip.

## Files

| File | Purpose |
|---|---|
| `distract.py` | Command-line interface. |
| `main.py` | `run_stuff`: the whole pipeline; reading review files. |
| `input.py`, `output.py` | Reading input; writing CSV / JS / longform / summary output. |
| `sentence_set.py` | Sentences, items and labels; distractor choice. |
| `wordfreq_distractor.py`, `distractor.py` | Candidate words and length/frequency thresholds. |
| `get_surprisal.py`, `backends/` | Surprisal from language models. |
| `limit_repeats.py`, `set_params.py`, `utils.py` | Repeat limits, parameters, punctuation handling. |
| `check_distractors.py`, `check-check-distractors.R` | Experimental LLM grammaticality check (not reliable yet), and exploratory plots of it. |
| `benchmark.py` | Compare models' run time on the same input. |

## Citing

There isn't yet a separate publication for this newer distractor generation. If you use it, please cite one of the A-maze papers, plus the specific language model you used to generate distractors:

- V. Boyce, R. Futrell, R. P. Levy (2020). Maze Made Easy: Better and easier measurement of incremental processing difficulty. *Journal of Memory and Language*.
- V. Boyce, R. P. Levy (2023). A-maze of Natural Stories: Comprehension and surprisal in the Maze task. *Glossa Psycholinguistics*.
