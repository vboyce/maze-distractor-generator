"""Hugging Face Transformers backends for surprisal computation"""
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import Union

# Elements in one batch of logits (rows x length x vocabulary): 1 GiB at float32.
_LOGIT_BUDGET = 2 ** 28


def detect_start_token_id(tokenizer) -> int:
    """The token a sequence starts with, read off the tokenizer's own behaviour.

    A tokenizer that prepends one token when asked for special tokens has named
    it. One that adds nothing (GPT-2, Pythia) leaves its BOS token, or failing
    that (Qwen) its EOS token, which is what separates documents in such a
    model's training data.

    Ported from Amani Maina-Kilaas's maze-distractor-generator
    (github.com/amanirmk/maze-distractor-generator, surprisal.py), GPL-3.
    """
    plain = tokenizer.encode("a", add_special_tokens=False)
    wrapped = tokenizer.encode("a", add_special_tokens=True)
    if wrapped == plain:
        if tokenizer.bos_token_id is not None:
            return tokenizer.bos_token_id
        if tokenizer.eos_token_id is not None:
            logging.warning("The tokenizer has no BOS token; starting sequences with its EOS token %r.",
                            tokenizer.eos_token)
            return tokenizer.eos_token_id
        raise ValueError("The tokenizer has neither a BOS nor an EOS token to start sequences with; "
                         "set prepend_start_token: False to score without one.")
    if wrapped[1:] == plain:
        return wrapped[0]
    raise ValueError(f"The tokenizer wraps text in special tokens other than one leading token "
                     f"({plain} became {wrapped}); set prepend_start_token: False to score without one.")


def check_float32(model, model_name: str):
    """Raise unless the model's weights are float32. Half precision puts
    surprisal on a grid coarser than the differences thresholds are compared at."""
    loaded = next(model.parameters()).dtype
    if loaded != torch.float32:
        raise ValueError(f"{model_name} loaded as {loaded}, not float32")


class _TransformersBase:
    """Shared tokenization and setup logic for Transformers backends"""

    def __init__(self, model_name: str, device: Union[str, torch.device] = None, revision: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    @property
    def model_commit(self):
        """The commit transformers resolved the model to on the Hugging Face
        Hub, for the run record; None for a model loaded from a local directory."""
        return self.model.config._commit_hash

    def _load_model(self, auto_class, model_name: str, revision: str):
        """Load the model as float32 (whatever the checkpoint declares), on self.device, in eval mode."""
        model = auto_class.from_pretrained(model_name, revision=revision, dtype=torch.float32)
        check_float32(model, model_name)
        model.to(self.device)
        model.eval()
        return model

    def _tokenize(self, prefix: str, word: str):
        """Strip, validate, tokenize prefix+word together, and return (prefix_tokens, word_tokens)."""
        prefix = prefix.strip()
        word = word.strip()
        if not prefix:
            raise ValueError("prefix must not be empty; surprisal requires at least one token of context")
        full_text = prefix + " " + word

        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt").to(self.device)
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False, return_tensors="pt").to(self.device)
        prefix_len = prefix_tokens.shape[1]

        if prefix_len > 0 and not torch.equal(prefix_tokens[0], full_tokens[0, :prefix_len]):
            prefix_list = prefix_tokens[0].tolist()
            full_list = full_tokens[0].tolist()
            if full_list[:prefix_len] != prefix_list:
                raise ValueError(
                    f"Prefix tokenization doesn't align with full text. "
                    f"Prefix tokens: {prefix_list}, Full start: {full_list[:prefix_len + 3]}"
                )

        word_tokens = full_tokens[:, prefix_len:]
        if word_tokens.shape[1] == 0:
            raise ValueError(f"Word '{word}' could not be tokenized")
        return prefix_tokens, word_tokens


class CausalLMBackend(_TransformersBase):
    """Surprisal backend for autoregressive / causal language models (GPT-2, GPT-Neo, Llama, etc.)

    With prepend_start_token (the default), the model reads the start token
    (see detect_start_token_id) before the prefix, as it saw at the start of a
    document in training. Otherwise the first token of the prefix is the first
    thing it reads.
    """

    def __init__(self, model_name: str, device: Union[str, torch.device] = None,
                 prepend_start_token: bool = True, revision: str = None):
        super().__init__(model_name, device, revision)
        self.model = self._load_model(AutoModelForCausalLM, model_name, revision)
        self.start_token_id = detect_start_token_id(self.tokenizer) if prepend_start_token else None
        self.max_positions = getattr(self.model.config, "max_position_embeddings", None)
        self._context_cache = {}  # prefix -> next-token log-probabilities after it

    def _context_tokens(self, prefix: str, word: str):
        """(context_tokens, word_tokens): the prefix's tokens, after the start
        token if there is one, and the word's. Raises if together they are
        longer than the model reads."""
        prefix_tokens, word_tokens = self._tokenize(prefix, word)
        if self.start_token_id is not None:
            start = torch.tensor([[self.start_token_id]], device=self.device)
            prefix_tokens = torch.cat([start, prefix_tokens], dim=1)
        n_tokens = prefix_tokens.shape[1] + word_tokens.shape[1]
        if self.max_positions is not None and n_tokens > self.max_positions:
            raise ValueError(f"{prefix!r} + {word!r} is {n_tokens} tokens, longer than the model reads "
                             f"({self.max_positions})")
        return prefix_tokens, word_tokens

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        prefix_tokens, word_tokens = self._context_tokens(prefix, word)
        log_prob_sum = 0.0
        current_prefix = prefix_tokens.clone()
        with torch.no_grad():
            for i, token_id in enumerate(word_tokens[0]):
                outputs = self.model(current_prefix)
                token_logits = outputs.logits[0, -1, :]
                token_log_probs = torch.log_softmax(token_logits, dim=-1)
                token_log_prob = token_log_probs[token_id].item()
                if not np.isfinite(token_log_prob):
                    token_str = self.tokenizer.decode([token_id.item()])
                    raise ValueError(
                        f"Model assigned zero probability to token {token_id.item()} ({repr(token_str)}) "
                        f"at position {i + 1} of word"
                    )
                log_prob_sum += token_log_prob
                current_prefix = torch.cat([current_prefix, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
        return -log_prob_sum / np.log(base)

    def _next_token_log_probs(self, prefix: str, context_tokens):
        """Log-probabilities of every vocabulary entry after the context, from
        one forward pass (the same pass get_surprisal makes for a word's first
        token). Kept for the most recent prefixes, which are reused across a
        label's chunks of candidates."""
        if prefix not in self._context_cache:
            if len(self._context_cache) >= 64:
                self._context_cache.clear()
            with torch.no_grad():
                logits = self.model(context_tokens).logits[0, -1, :]
            self._context_cache[prefix] = torch.log_softmax(logits, dim=-1)
        return self._context_cache[prefix]

    def score_candidates(self, prefix: str, candidates, exact_below: float, base: float = 2.0):
        """Surprisal of each candidate after prefix, as a list of
        (surprisal, exact) pairs.

        Ported, in outline, from Amani Maina-Kilaas's maze-distractor-generator
        (github.com/amanirmk/maze-distractor-generator, surprisal.py), GPL-3.

        One forward pass after the prefix scores every candidate's first token.
        A single-token candidate's surprisal is then exact. A word's surprisal
        is at least its first token's, so if that already reaches exact_below,
        the first token's surprisal is returned as a lower bound (exact=False)
        and the word's other tokens are never run. The remaining multi-token
        candidates are scored exactly, in batches. Batched scores can differ
        from get_surprisal's in the third decimal place (up to ~1.1e-3 bits on
        Pythia-160m), which is far finer than surprisal needs to be here. Pass
        exact_below=inf for exact scores throughout.
        """
        results = [None] * len(candidates)
        pending = []  # (candidate index, context + word token ids, number of context tokens)
        for k, word in enumerate(candidates):
            context_tokens, word_tokens = self._context_tokens(prefix, word)
            first_id = word_tokens[0, 0]
            first_log_prob = self._next_token_log_probs(prefix, context_tokens)[first_id].item()
            if not np.isfinite(first_log_prob):
                raise ValueError(f"Model assigned zero probability to the first token of {word!r}")
            first = -(0.0 + first_log_prob) / np.log(base)  # as get_surprisal computes it
            if word_tokens.shape[1] == 1:
                results[k] = (first, True)
            elif first >= exact_below:
                results[k] = (first, False)
            else:
                token_ids = context_tokens[0].tolist() + word_tokens[0].tolist()
                pending.append((k, token_ids, context_tokens.shape[1]))
        for (k, _, _), surprisal in zip(pending, self._score_suffixes(pending, base)):
            results[k] = (surprisal, True)
        return results

    def _score_suffixes(self, pending, base: float):
        """For each (index, token_ids, n_context) in pending, the summed
        surprisal of token_ids[n_context:] given the tokens before them.

        Sequences are batched only with others of the same length, so no
        padding or attention mask is needed."""
        totals = [None] * len(pending)
        by_length = {}
        for i, (_, token_ids, _) in enumerate(pending):
            by_length.setdefault(len(token_ids), []).append(i)
        vocab_size = self.model.config.vocab_size
        for width, indices in by_length.items():
            batch_size = max(1, _LOGIT_BUDGET // (width * vocab_size))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                input_ids = torch.tensor([pending[i][1] for i in batch], device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids).logits
                for row, i in enumerate(batch):
                    _, token_ids, n_context = pending[i]
                    # Position p predicts token p + 1.
                    log_probs = torch.log_softmax(logits[row, n_context - 1:width - 1], dim=-1)
                    suffix = torch.tensor(token_ids[n_context:], device=log_probs.device)
                    token_log_probs = log_probs[torch.arange(len(suffix)), suffix]
                    if not torch.isfinite(token_log_probs).all():
                        raise ValueError(f"Model assigned zero probability to a token of candidate {i}")
                    totals[i] = -token_log_probs.sum().item() / np.log(base)
        return totals


class MaskedLMBackend(_TransformersBase):
    """Surprisal backend for masked language models (BERT, RoBERTa, etc.)
    Uses left-to-right iterative unmasking for multi-token words.

    The input always starts with the tokenizer's [CLS] token (if it has one), so
    prepend_start_token must be True; it is accepted so that every backend takes
    the same parameters."""

    def __init__(self, model_name: str, device: Union[str, torch.device] = None,
                 prepend_start_token: bool = True, revision: str = None):
        if not prepend_start_token:
            raise ValueError("The masked-LM backend always starts its input with [CLS]; "
                             "prepend_start_token: False is not supported")
        super().__init__(model_name, device, revision)
        self.model = self._load_model(AutoModelForMaskedLM, model_name, revision)
        if self.tokenizer.mask_token is None:
            raise ValueError(f"Tokenizer for {model_name} does not have a mask token")

    def _build_masked_input(self, prefix_tokens, num_masks):
        """Build [CLS] prefix [MASK]*num_masks [SEP] tensor."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        ids = []
        if cls_id is not None:
            ids.append(cls_id)
        ids.extend(prefix_tokens[0].tolist())
        mask_start = len(ids)
        ids.extend([mask_id] * num_masks)
        if sep_id is not None:
            ids.append(sep_id)
        return torch.tensor([ids], device=self.device), mask_start

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        prefix_tokens, word_tokens = self._tokenize(prefix, word)
        n_word_tokens = word_tokens.shape[1]
        if n_word_tokens > 1:
            token_strs = [self.tokenizer.decode([t]) for t in word_tokens[0]]
            logging.info(
                "Word '%s' is %d tokens %s; using iterative unmasking",
                word.strip(), n_word_tokens, token_strs,
            )

        input_ids, mask_start = self._build_masked_input(prefix_tokens, n_word_tokens)
        log_prob_sum = 0.0
        with torch.no_grad():
            for i, token_id in enumerate(word_tokens[0]):
                outputs = self.model(input_ids)
                mask_pos = mask_start + i
                token_log_probs = torch.log_softmax(outputs.logits[0, mask_pos, :], dim=-1)
                token_log_prob = token_log_probs[token_id].item()
                if not np.isfinite(token_log_prob):
                    token_str = self.tokenizer.decode([token_id.item()])
                    raise ValueError(
                        f"Model assigned zero probability to token {token_id.item()} ({repr(token_str)}) "
                        f"at mask position {i + 1} of word"
                    )
                log_prob_sum += token_log_prob
                input_ids[0, mask_pos] = token_id
        return -log_prob_sum / np.log(base)
