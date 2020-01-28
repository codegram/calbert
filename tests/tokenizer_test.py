import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

import torch
from tokenizers import Encoding
from omegaconf import OmegaConf

from calbert import tokenizer

from .conftest import InputData, folder


def train_tokenizer(
    input_file_and_outdir: (str, str)
) -> (tokenizer.CalbertTokenizer, str):
    input_file, outdir = input_file_and_outdir
    args, cfg = tokenizer_args_and_cfg(input_file, outdir)

    return tokenizer.train(args, cfg), outdir


def tokenizer_args_and_cfg(
    input_file: str, outdir: str
) -> (argparse.Namespace, OmegaConf):
    args = tokenizer.arguments().parse_args(
        ["--input-file", input_file, "--out-dir", outdir]
    )
    config = [
        "vocab.max_size=10",
        "vocab.min_frequency=2",
        "vocab.lowercase=True",
        "training.max_seq_length=12",
    ]
    cfg = OmegaConf.from_dotlist(config)
    return args, cfg


@pytest.fixture(scope="module")
def input_file_and_outdir(which="train") -> (str, str):
    with InputData("train") as train_file:
        with folder() as outdir:
            yield train_file, outdir


def encoding_to_tensor(e: Encoding) -> torch.Tensor:
    return torch.stack(
        [
            torch.tensor(e.ids),
            torch.tensor(e.special_tokens_mask),
            torch.tensor(e.attention_mask),
            torch.tensor(e.type_ids),
        ]
    )


@pytest.mark.describe("tokenizer.CalbertTokenizer")
class TestCalbertokenizer:
    @pytest.mark.it("Trains a tokenizer on some corpus")
    def test_train(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        assert len(t) == 35
        assert t.id_to_token(0) == "<unk>"
        assert t.id_to_token(1) == "<pad>"
        assert t.id_to_token(2) == "[MASK]"
        assert t.id_to_token(3) == "[SEP]"
        assert t.id_to_token(4) == "[CLS]"

    @pytest.mark.it("Encodes single sentences BERT-style with CLS and SEP")
    def test_single_sentence_encoding(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)
        encoded = t.encode("hOla com anem")
        assert len(encoded.ids) == 12
        assert t.id_to_token(encoded.ids[0]) == "[CLS]"
        assert t.id_to_token(encoded.ids[-1]) == "[SEP]"

    @pytest.mark.it("Encodes pairs of sentences BERT-style with CLS and SEP")
    def test_sentence_pair_encoding(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)
        encoded = t.encode("hola com anem", "molt be i tu")
        assert len(encoded.ids) == 12
        assert t.id_to_token(encoded.ids[0]) == "[CLS]"
        assert t.id_to_token(encoded.ids[6]) == "[SEP]"
        assert t.id_to_token(encoded.ids[-1]) == "[SEP]"

    @pytest.mark.it(
        "Encodes pairs of sentences BERT-style with CLS and SEP padding correctly"
    )
    def test_sentence_pair_encoding_padding(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)
        encoded = t.encode("com", "deu")
        assert len(encoded.ids) == 12
        assert t.id_to_token(encoded.ids[0]) == "[CLS]"
        assert t.id_to_token(encoded.ids[5]) == "[SEP]"
        assert t.id_to_token(encoded.ids[-2]) == "[SEP]"
        assert t.id_to_token(encoded.ids[-1]) == "<pad>"

    @pytest.mark.it(
        "Truncates encodings to the max sequence length, respecting special tokens"
    )
    def test_truncates_encodings(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        encoded = t.encode("Hola com Anem")
        assert len(encoded.tokens) == 12
        assert encoded.tokens == [
            "[CLS]",
            "▁",
            "h",
            "o",
            "l",
            "a",
            "▁",
            "c",
            "o",
            "m",
            "▁",
            "[SEP]",
        ]

    @pytest.mark.it("Pads encodings up to the max sequence length")
    def test_pads_encodings(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        encoded = t.encode("hola")
        assert encoded.tokens == [
            "[CLS]",
            "▁",
            "h",
            "o",
            "l",
            "a",
            "[SEP]",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
            "<pad>",
        ]

    @pytest.mark.it("Encodes a batch of sentences")
    def test_batch_encoding(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)
        first, second = t.encode_batch(["hola", "genial"])
        assert len(first.tokens) == 12
        assert len(second.tokens) == 12

        first, second = t.encode_batch(
            ["hola, no entenc be per que aquesta frase", "seria tan llarga"]
        )
        assert len(first.tokens) == 12
        assert len(second.tokens) == 12

    @pytest.mark.it("Encodes a batch of sentence pairs")
    def test_batch_encoding_pairs(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)
        pairs = t.encode_batch(
            [("hola, no entenc be per que aquesta frase", "seria tan llarga")]
        )
        assert len(pairs[0].tokens) == 12

    @pytest.mark.it("Saves the tokenizer's vocab and merges")
    def test_saves_tokenizer(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        got = list(glob.glob(outdir + "/*"))
        got.sort()
        expected = [
            outdir + f"/ca.bpe.uncased.{len(t)}-vocab.json",
            outdir + f"/ca.bpe.uncased.{len(t)}-merges.txt",
        ]
        expected.sort()
        assert got == expected
