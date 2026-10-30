# Word lists for English-only heuristics

These lists come from Amani Maina-Kilaas's version of this generator
(<https://github.com/amanirmk/maze-distractor-generator>, commit `24e1596`,
`src/maze_distractors/data/`), also under GPL-3. They are copied unchanged.
The scripts that build them are in that repository's `scripts/` folder; they
need SUBTLEX-US, and are not ported here.

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

Only 31 of the names and 2 of the abbreviations are on `curated_word_list.txt`,
so with the default word list this leaves out few words; it matters more for
your own `include_words` list.
