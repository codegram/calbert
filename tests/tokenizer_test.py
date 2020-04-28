import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

from transformers import AlbertTokenizer
from omegaconf import OmegaConf

from calbert.tokenizer import arguments, train, load

from .conftest import InputData, folder


def train_tokenizer(in_and_out: (str, str)) -> (AlbertTokenizer, str):
    input_file, outdir = in_and_out
    args, cfg = tokenizer_args_and_cfg(input_file, outdir)

    prefix = train(args, cfg)
    return load(cfg, Path(f"{prefix}.model")), prefix


def tokenizer_args_and_cfg(
    input_file: str, outdir: str
) -> (argparse.Namespace, OmegaConf):
    args = arguments().parse_args(["--input-file", input_file, "--out-dir", outdir])
    config = [
        "vocab.max_size=44",
        "vocab.lowercase=True",
        "training.max_seq_length=12",
    ]
    cfg = OmegaConf.from_dotlist(config)
    return args, cfg


@pytest.fixture(scope="module")
def tokenizer(which="train") -> (str, str):
    with InputData(which) as train_file:
        with folder() as outdir:
            yield train_tokenizer((train_file, outdir))[0]


@pytest.fixture(scope="module")
def input_file_and_outdir(which="train") -> (str, str):
    with InputData(which) as train_file:
        with folder() as outdir:
            yield train_file, outdir


@pytest.mark.describe("tokenizer")
class TestTokenizer:
    @pytest.mark.it("Trains a tokenizer on some corpus")
    def test_train(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        tokens = t.tokenize("Hola, com anem? Tot bé?")
        assert tokens == [
            "▁",
            "h",
            "o",
            "l",
            "a",
            ",",
            "▁",
            "c",
            "o",
            "m",
            "▁",
            "a",
            "n",
            "e",
            "m",
            "?",
            "▁",
            "t",
            "o",
            "t",
            "▁",
            "b",
            "é?",
        ]

        assert len(t) == 44
        assert t._convert_token_to_id("<pad>") == 0
        assert t._convert_token_to_id("<unk>") == 1
        assert t._convert_token_to_id("[CLS]") == 2
        assert t._convert_token_to_id("[SEP]") == 3
        assert t._convert_token_to_id("[MASK]") == 4

    @pytest.mark.it("Encodes single sentences BERT-style with CLS and SEP")
    def test_sequence_builders(self, tokenizer):
        text = tokenizer.encode("Hola, com anem?")
        text_2 = tokenizer.encode("Tot bé?")

        encoded_sentence = tokenizer.build_inputs_with_special_tokens(text)
        encoded_pair = tokenizer.build_inputs_with_special_tokens(text, text_2)

        assert encoded_sentence == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id
        ]
        assert encoded_pair == [tokenizer.cls_token_id] + text + [
            tokenizer.sep_token_id
        ] + text_2 + [tokenizer.sep_token_id]

    @pytest.mark.it("Saves the tokenizer's vocab and model")
    def test_saves_tokenizer(self, input_file_and_outdir):
        t, outdir = train_tokenizer(input_file_and_outdir)

        got = list(glob.glob(outdir + "*"))
        got.sort()
        expected = [
            outdir + f".vocab",
            outdir + f".model",
        ]
        expected.sort()
        assert got == expected
