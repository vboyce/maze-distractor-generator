# maze-distractor-generator

Generates distractor words for A-maze reading experiments, using a language
model's surprisal. A library and CLI that other studies call (e.g.
`kid-maze-passages/stimuli-build/generate_distractors.py`).

## Folder types

Type definitions are in `~/.claude/project-types/`. Each folder below has a
CLAUDE.md that imports its type. The type is set by the file being edited, not
by the task.

| Folder | Type |
|---|---|
| repo root (`*.py`, `params.txt`, `check-check-distractors.R`) | reusable |
| `backends/`, `tests/`, `scripts/` | reusable |
| `input/` | readonly (example inputs used by the README, `distract.py --help` and `benchmark.py`) |
| `word_lists/` | readonly **once step 7c has added its list** (see below) |
| `script/` | test runners (`script/test`, `script/test_fast`); edit only when asked |

`word_lists/` is not locked yet: step 7c of the fixes plan adds
`noun_phrase_breakers.txt` and its provenance to `word_lists/README.md`. After
that, add `word_lists/CLAUDE.md` (importing the readonly type) and a `deny`
rule for it in `.claude/settings.json`.

Readonly folders are enforced by `deny` rules in `.claude/settings.json`. Those
rules only load when the session is started in this repo's root.

The rest of this file applies to the repo root.

@~/.claude/project-types/reusable.md
