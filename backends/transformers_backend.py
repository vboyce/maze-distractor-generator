"""Hugging Face Transformers backends for surprisal computation"""
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import Union


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

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        prefix_tokens, word_tokens = self._tokenize(prefix, word)
        if self.start_token_id is not None:
            start = torch.tensor([[self.start_token_id]], device=self.device)
            prefix_tokens = torch.cat([start, prefix_tokens], dim=1)
        n_tokens = prefix_tokens.shape[1] + word_tokens.shape[1]
        if self.max_positions is not None and n_tokens > self.max_positions:
            raise ValueError(f"{prefix!r} + {word!r} is {n_tokens} tokens, longer than the model reads "
                             f"({self.max_positions})")
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
