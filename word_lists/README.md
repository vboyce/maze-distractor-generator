# Word lists

## Candidates and exclusions

- `curated_word_list.txt`: the default `include_words` for English, the
  candidate distractors (see "Word lists" in the main README for how it was
  built; `scripts/curate_wordlist.py` was one step).
- `exclude.txt`: the default `exclude_words`, words never used as distractors.

## English-only heuristics

These lists come from Amani Maina-Kilaas's version of this generator
(<https://github.com/amanirmk/maze-distractor-generator>, commit `24e1596`,
`src/maze_distractors/data/`), also under GPL-3. They are copied unchanged.
The scripts that build them are in that repository's `scripts/` folder; they
need SUBTLEX-US (or, for `noun_phrase_breakers.txt`, WordNet and
lemminflect), and are not ported here.

They are used only when `language` is `"en"`.

- `first_names.txt` (842 words), left out with `avoid_names_and_abbreviations`:
  every first name in the 1990 US Census lists that is capitalized at least
  95% of the time in SUBTLEX-US (Brysbaert & New, 2009) and occurs in
  lowercase less than once per million words. Words that are also ordinary
  words, like "sue" and "mark", are not on it.
- `abbreviations.txt` (54 words), left out with `avoid_names_and_abbreviations`:
  every USPS state or territory code, and every abbreviated title in the US
  Government Publishing Office's *Style Manual*, that is capitalized at least
  50% of the time in SUBTLEX-US and occurs in lowercase less than 10 per
  million words.
- `often_capitalized.txt` (703 words), with `check_often_capitalized`: words
  that must meet their surprisal target capitalized too, since a reader may
  take them for a name, place or title ("grace", "oxford", "academy"). A word
  is on it if SUBTLEX-US has it capitalized at least 40% of the time and
  `gpt2-medium` prefers it capitalized mid-sentence by at least 3 bits (6
  bits, for a word SUBTLEX-US lacks). Only words of `curated_word_list.txt`
  were checked, so every word on it is on `curated_word_list.txt`, but a word
  you add to that list, or a word of your own `include_words` list, is not
  checked. To cover new words, rebuild it with
  `scripts/build_often_capitalized_list.py` in Amani's repository.
- `noun_phrase_breakers.txt` (5,079 words), with `break_noun_phrases`: the
  only candidates straight after a determiner, since after "the" a noun,
  adjective, participle or adverb could go on to make a sentence ("The
  alluded..."), however surprising. A word is on it if it is a closed-class
  word with no use as a noun, adjective or adverb (pronouns, prepositions,
  conjunctions, ...), or if WordNet gives it no noun, adjective or adverb
  sense, it is not a participle, and it is a finite form of a WordNet verb
  that lemminflect also derives from that verb. What is left is mostly
  finite verbs and closed-class words. Every lowercase English word wordfreq
  knows was checked, not only `curated_word_list.txt`; 1,099 of the words
  are on `curated_word_list.txt`. WordNet's rare senses make the rule
  strict: it leaves out some words that could be used. Rebuild it with
  `scripts/build_noun_phrase_breakers.py` in Amani's repository.

Only 31 of the names and 2 of the abbreviations are on `curated_word_list.txt`,
so with the default word list this leaves out few words; it matters more for
your own `include_words` list.
