"""A tiny randomly initialized GPT-2 (and a tiny BERT) with a word-level
tokenizer, saved to a directory, so backend tests can load a real model
without a download.

The tokenizer splits on whitespace and has one token per vocabulary word
(anything else is [UNK]). Like GPT-2's, it adds no special tokens itself and
uses <|endoftext|> as both BOS and EOS.
"""
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

VOCAB = ["<|endoftext|>", "[UNK]", "The", "A", "cat", "dog", "sat", "ran", "sat.", "ran."]
MASKED_VOCAB = VOCAB + ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]
MAX_POSITIONS = 16


def make_tokenizer(vocab=VOCAB, **special_tokens):
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(vocab)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", **special_tokens)


def save_tiny_lm(directory, dtype=torch.float32):
    """Save the tokenizer and a seeded random model to directory; return it."""
    make_tokenizer(bos_token="<|endoftext|>", eos_token="<|endoftext|>").save_pretrained(directory)
    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(VOCAB), n_embd=16, n_layer=1, n_head=2, n_positions=MAX_POSITIONS)
    GPT2LMHeadModel(config).to(dtype).save_pretrained(directory)
    return str(directory)


def save_tiny_masked_lm(directory, tokenizer_max_length=None):
    """Save a word-level tokenizer with BERT's special tokens and a seeded
    random BERT to directory; return it. tokenizer_max_length, if given, is
    the tokenizer's model_max_length (RoBERTa's is 2 below its config's
    max_position_embeddings)."""
    tokenizer_settings = {} if tokenizer_max_length is None else {"model_max_length": tokenizer_max_length}
    make_tokenizer(MASKED_VOCAB, cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                   pad_token="[PAD]", **tokenizer_settings).save_pretrained(directory)
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(MASKED_VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                        intermediate_size=32, max_position_embeddings=MAX_POSITIONS)
    BertForMaskedLM(config).save_pretrained(directory)
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
