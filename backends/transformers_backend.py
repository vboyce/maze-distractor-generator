"""Hugging Face Transformers backends for surprisal computation"""
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import Union


class _TransformersBase:
    """Shared tokenization and setup logic for Transformers backends"""

    def __init__(self, model_name: str, device: Union[str, torch.device] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    """Surprisal backend for autoregressive / causal language models (GPT-2, GPT-Neo, Llama, etc.)"""

    def __init__(self, model_name: str, device: Union[str, torch.device] = None):
        super().__init__(model_name, device)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_surprisal(self, prefix: str, word: str, base: float = 2.0) -> float:
        prefix_tokens, word_tokens = self._tokenize(prefix, word)
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
    Uses left-to-right iterative unmasking for multi-token words."""

    def __init__(self, model_name: str, device: Union[str, torch.device] = None):
        super().__init__(model_name, device)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
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
