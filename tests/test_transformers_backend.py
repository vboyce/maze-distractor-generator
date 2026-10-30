"""The transformers backends, on a tiny random model (see tests/tiny_lm.py)."""
import sys
import os
import csv

import pytest
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import run_stuff
from backends import load_surprisal_model
from backends.transformers_backend import CausalLMBackend, detect_start_token_id, check_float32
from tests.tiny_lm import VOCAB, MAX_POSITIONS, hand_surprisal, make_tokenizer

REPO = os.path.dirname(os.path.dirname(__file__))
BOS = VOCAB.index("<|endoftext|>")


def ids(*words):
    return [VOCAB.index(w) for w in words]


@pytest.fixture(scope="module")
def tiny_model(tiny_lm_dir):
    return AutoModelForCausalLM.from_pretrained(tiny_lm_dir).eval()


def real_word_surprisals(tmp_path, model_dir, **params):
    infile = tmp_path / "in.csv"
    infile.write_text("type,item_num,sentence\nx,1,The cat sat.\n")
    logfile = tmp_path / "log.csv"
    run_stuff(str(infile), str(tmp_path / "out.csv"), logfile=str(logfile),
              parameters={"model": model_dir, "num_to_test": 3, "max_repeat": 0,
                          "include_words": os.path.join(REPO, "curated_word_list.txt"),
                          "exclude_words": os.path.join(REPO, "exclude.txt"), **params})
    with open(logfile, newline="") as f:
        return {row["word"]: float(row["actual_surprisal"])
                for row in csv.DictReader(f) if row["record_type"] == "real_word"}


class TestStartToken:
    def test_run_scores_words_after_the_start_token(self, tmp_path, tiny_lm_dir, tiny_model):
        surprisals = real_word_surprisals(tmp_path, tiny_lm_dir)
        assert surprisals["cat"] == pytest.approx(hand_surprisal(tiny_model, ids("<|endoftext|>", "The"), ids("cat")))
        assert surprisals["sat."] == pytest.approx(
            hand_surprisal(tiny_model, ids("<|endoftext|>", "The", "cat"), ids("sat.")))

    def test_run_without_the_start_token_scores_after_the_bare_prefix(self, tmp_path, tiny_lm_dir, tiny_model):
        surprisals = real_word_surprisals(tmp_path, tiny_lm_dir, prepend_start_token=False)
        assert surprisals["cat"] == pytest.approx(hand_surprisal(tiny_model, ids("The"), ids("cat")))


class TestDetectStartTokenId:
    def test_tokenizer_that_adds_nothing_starts_with_bos(self):
        tokenizer = make_tokenizer(bos_token="A", eos_token="<|endoftext|>")
        assert detect_start_token_id(tokenizer) == VOCAB.index("A")

    def test_tokenizer_without_bos_starts_with_eos_and_warns(self, caplog):
        tokenizer = make_tokenizer(eos_token="<|endoftext|>")
        assert detect_start_token_id(tokenizer) == BOS
        assert "EOS" in caplog.text

    def test_tokenizer_without_bos_or_eos_raises(self):
        with pytest.raises(ValueError, match="neither a BOS nor an EOS"):
            detect_start_token_id(make_tokenizer())

    def test_tokenizer_that_adds_a_leading_token_starts_with_it(self):
        from tokenizers import processors
        tokenizer = make_tokenizer(bos_token="A")
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="The $A", special_tokens=[("The", VOCAB.index("The"))])
        assert detect_start_token_id(tokenizer) == VOCAB.index("The")

    def test_tokenizer_that_wraps_text_on_both_sides_raises(self):
        from tokenizers import processors
        tokenizer = make_tokenizer(bos_token="A")
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="The $A A", special_tokens=[("The", VOCAB.index("The")), ("A", VOCAB.index("A"))])
        with pytest.raises(ValueError, match="other than one leading token"):
            detect_start_token_id(tokenizer)


class TestCausalLMBackend:
    def test_multi_token_word_is_scored_after_the_start_token(self, tiny_lm_dir, tiny_model):
        backend = CausalLMBackend(tiny_lm_dir)
        expected = hand_surprisal(tiny_model, ids("<|endoftext|>", "The"), ids("cat", "sat."))
        assert backend.get_surprisal("The", "cat sat.") == pytest.approx(expected)

    def test_without_the_start_token_scores_after_the_bare_prefix(self, tiny_lm_dir, tiny_model):
        backend = CausalLMBackend(tiny_lm_dir, prepend_start_token=False)
        expected = hand_surprisal(tiny_model, ids("The"), ids("cat", "sat."))
        assert backend.get_surprisal("The", "cat sat.") == pytest.approx(expected)

    def test_text_longer_than_the_model_reads_raises(self, tiny_lm_dir):
        backend = CausalLMBackend(tiny_lm_dir)
        prefix = " ".join(["cat"] * MAX_POSITIONS)
        with pytest.raises(ValueError, match="longer than the model reads"):
            backend.get_surprisal(prefix, "sat.")

    def test_text_that_fits_only_without_the_start_token_raises_with_it(self, tiny_lm_dir):
        prefix = " ".join(["cat"] * (MAX_POSITIONS - 1))
        CausalLMBackend(tiny_lm_dir, prepend_start_token=False).get_surprisal(prefix, "sat.")
        with pytest.raises(ValueError, match="longer than the model reads"):
            CausalLMBackend(tiny_lm_dir).get_surprisal(prefix, "sat.")

    def test_half_precision_checkpoint_loads_as_float32(self, tmp_path):
        import torch
        from tests.tiny_lm import save_tiny_lm
        backend = CausalLMBackend(save_tiny_lm(tmp_path, dtype=torch.float16))
        assert next(backend.model.parameters()).dtype == torch.float32


def test_check_float32_rejects_a_half_precision_model(tiny_model):
    import copy
    import torch
    with pytest.raises(ValueError, match="float32"):
        check_float32(copy.deepcopy(tiny_model).to(torch.float16), "tiny")


class TestStartTokenOnOtherBackends:
    def test_masked_lm_cannot_turn_the_start_token_off(self):
        from backends.transformers_backend import MaskedLMBackend
        with pytest.raises(ValueError, match="prepend_start_token"):
            MaskedLMBackend("bert-base-uncased", prepend_start_token=False)

    def test_litellm_cannot_prepend_a_start_token(self):
        pytest.importorskip("litellm")
        with pytest.raises(ValueError, match="prepend_start_token"):
            load_surprisal_model("openai/some-model", backend="litellm", prepend_start_token=True)

    def test_litellm_rejects_a_revision(self):
        pytest.importorskip("litellm")
        with pytest.raises(ValueError, match="revision"):
            load_surprisal_model("openai/some-model", backend="litellm",
                                 prepend_start_token=False, revision="abc123")
