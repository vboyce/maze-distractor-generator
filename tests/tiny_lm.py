"""A tiny randomly initialized GPT-2 and word-level tokenizer, saved to a
directory, so backend tests can load a real model without a download.

The tokenizer splits on whitespace and has one token per vocabulary word
(anything else is [UNK]). Like GPT-2's, it adds no special tokens itself and
uses <|endoftext|> as both BOS and EOS.
"""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

VOCAB = ["<|endoftext|>", "[UNK]", "The", "A", "cat", "dog", "sat", "ran", "sat.", "ran."]
MAX_POSITIONS = 16


def make_tokenizer(**special_tokens):
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", **special_tokens)


def save_tiny_lm(directory, dtype=torch.float32):
    """Save the tokenizer and a seeded random model to directory; return it."""
    make_tokenizer(bos_token="<|endoftext|>", eos_token="<|endoftext|>").save_pretrained(directory)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(VOCAB), n_embd=16, n_layer=1, n_head=2, n_positions=MAX_POSITIONS)
    GPT2LMHeadModel(config).to(dtype).save_pretrained(directory)
    return str(directory)


def hand_surprisal(model, context_ids, word_ids):
    """Surprisal in bits of word_ids after context_ids, from the model's logits,
    one token at a time."""
    total = 0.0
    ids = list(context_ids)
    with torch.no_grad():
        for token_id in word_ids:
            logits = model(torch.tensor([ids])).logits[0, -1]
            total -= torch.log_softmax(logits, dim=-1)[token_id].item()
            ids.append(token_id)
    return total / torch.log(torch.tensor(2.0)).item()
