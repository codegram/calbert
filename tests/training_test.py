import logging
import os
import re
import tempfile
import glob
import argparse
import pytest
from pathlib import Path

from omegaconf import OmegaConf
import torch

from calbert import dataset, training, tokenizer
from calbert.dataset import Tokenize, SentencePair, mask_tokens
from calbert.model import CalbertForMaskedLM
from transformers import AlbertForMaskedLM

from .tokenizer_test import train_tokenizer
from .conftest import InputData, folder


@pytest.fixture(scope="module")
def training_args_cfg():
    with InputData("train") as train_file:
        with InputData("valid") as valid_file:
            with folder() as tokenizer_dir:
                tok, prefix = train_tokenizer((train_file, tokenizer_dir))

                training_args = training.arguments().parse_args(
                    [
                        "--tokenizer-path",
                        f"{prefix}.model",
                        "--train-path",
                        str(train_file),
                        "--valid-path",
                        str(valid_file),
                        "--train-batch-size",
                        "1",
                        "--eval-batch-size",
                        "1",
                        "--epochs",
                        "1",
                        "--max-items",
                        "1",
                    ]
                )

                training_config = [
                    "training.max_seq_length=4",
                    "training.masked_lm_prob=0.1",
                    "training.weight_decay=0.0",
                    "training.learning_rate=5e-05",
                    "seed=42",
                    "model.name=test",
                    "model.hidden_size=312",
                    "model.embedding_size=64",
                    "model.initializer_range=0.02",
                    "model.intermediate_size=312",
                    "model.max_position_embeddings=128",
                    "model.num_attention_heads=4",
                    "vocab.lowercase=False",
                    "vocab.max_size=10",
                ]

                training_cfg = OmegaConf.from_dotlist(training_config)

                yield training_args, training_cfg, tok


@pytest.mark.describe("training.train")
class TestTraining:
    @pytest.mark.it("Trains the model")
    def test_process(self, training_args_cfg):
        args, cfg, tok = training_args_cfg

        learn = training.train(args, cfg, test_mode=True)

        model = learn.model

        tokenize = Tokenize(tok, max_seq_len=cfg.training.max_seq_length)

        (token_ids, attention_mask, type_ids) = tokenize(
            SentencePair("Hola com anem?", "Molt bé i tu?"),
        )

        masked_token_ids, labels = mask_tokens(
            token_ids,
            tok=tok,
            ignore_index=dataset.IGNORE_INDEX,
            probability=cfg.training.masked_lm_prob,
        )

        batch_inputs = masked_token_ids.unsqueeze(0)

        model.__class__ = AlbertForMaskedLM

        predictions = model(batch_inputs, token_type_ids=type_ids.unsqueeze(0),)[0][0]

        assert predictions.shape == (cfg.training.max_seq_length, len(tok))

        model.__class__ = CalbertForMaskedLM

        learn.validate()

        perplexity = learn.metrics[0].value.item()

        assert perplexity > 0
