import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from calbert import dataset, utils

from .tokenizer_test import train_tokenizer
from .conftest import InputData, folder


@pytest.fixture(scope="module")
def dataset_args_cfg():
    with InputData("train") as train_file:
        with InputData("valid") as valid_file:
            with folder() as tokenizer_dir:
                with folder() as dataset_dir:
                    tokenizer, _ = train_tokenizer((train_file, tokenizer_dir))
                    args = dataset.arguments().parse_args(
                        [
                            "--tokenizer-dir",
                            tokenizer_dir,
                            "--out-dir",
                            dataset_dir,
                            "--train-file",
                            train_file,
                            "--valid-file",
                            valid_file,
                        ]
                    )
                    config = [
                        "training.max_seq_length=12",
                        "vocab.max_size=10",
                    ]
                    cfg = OmegaConf.from_dotlist(config)
                    yield args, cfg, tokenizer


@pytest.mark.describe("dataset.Dataset")
class TestDataset:
    @pytest.mark.it("Turns raw text into a dataset consumable with random access")
    def test_process(self, dataset_args_cfg):
        args, cfg, tokenizer = dataset_args_cfg
        outdir = args.out_dir
        dataset.process(args, cfg)
        train_ds = dataset.CalbertDataset(
            outdir,
            split="train",
            max_seq_length=cfg.training.max_seq_length,
            max_vocab_size=cfg.vocab.max_size,
        )
        valid_ds = dataset.CalbertDataset(
            outdir,
            split="valid",
            max_seq_length=cfg.training.max_seq_length,
            max_vocab_size=cfg.vocab.max_size,
        )
        assert len(train_ds) == 4
        assert len(valid_ds) == 1

        tensor = train_ds[3]
        assert tensor.shape == (4, 12)
        assert [tokenizer.id_to_token(x) for x in tensor[0].tolist()] == [
            "[CLS]",
            "▁",
            "H",
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
