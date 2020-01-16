import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from calbert import dataset, training, tokenizer

from .tokenizer_test import train_tokenizer, encoding_to_tensor
from .conftest import InputData, folder


@pytest.fixture(scope="module")
def training_args_cfg():
    with InputData("train") as train_file:
        with InputData("valid") as valid_file:
            with folder() as tokenizer_dir:
                with folder() as dataset_dir:
                    with folder() as model_dir:
                        with folder() as tensorboard_dir:
                            tok, _ = train_tokenizer((train_file, tokenizer_dir))
                            dataset_args = dataset.arguments().parse_args(
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
                            dataset_config = [
                                "training.max_seq_length=12",
                                "vocab.max_size=10",
                            ]
                            dataset_cfg = OmegaConf.from_dotlist(dataset_config)

                            dataset.process(
                                dataset_args, dataset_cfg,
                            )

                            training_args = training.arguments().parse_args(
                                [
                                    "--tokenizer-dir",
                                    tokenizer_dir,
                                    "--out-dir",
                                    model_dir,
                                    "--dataset-dir",
                                    dataset_dir,
                                    "--tensorboard-dir",
                                    tensorboard_dir,
                                    "--per_gpu_train_batch_size",
                                    "2",
                                    "--per_gpu_eval_batch_size",
                                    "2",
                                ]
                            )

                            training_config = [
                                "training.max_seq_length=12",
                                "training.masked_lm_prob=0.1",
                                "training.num_train_steps=1",
                                "training.num_warmup_steps=0",
                                "training.epochs=1",
                                "training.weight_decay=0.0",
                                "training.learning_rate=5e-05",
                                "training.adam_epsilon=1e-8",
                                "training.max_grad_norm=1.0",
                                "seed=42",
                                "model.name=test",
                                "vocab.lowercase=True",
                                "vocab.max_size=10",
                            ]

                            training_cfg = OmegaConf.from_dotlist(training_config)

                            yield training_args, training_cfg, model_dir, tensorboard_dir, tok


@pytest.mark.describe("training.train")
class TestTraining:
    @pytest.mark.it("Trains the model")
    def test_process(self, training_args_cfg):
        args, cfg, model_path, tensorboard_dir, tok = training_args_cfg

        args.wandb = False

        model = training.train(args, cfg)

        encoding = tok.process("Hola com anem")
        (
            token_ids,
            special_tokens_mask,
            attention_mask,
            type_ids,
        ) = encoding_to_tensor(encoding)

        masked_token_ids, labels = training.mask_tokens(
            token_ids, special_tokens_mask=special_tokens_mask, tokenizer=tok, cfg=cfg,
        )

        batch_inputs = masked_token_ids.unsqueeze(0)

        predictions = model(batch_inputs, token_type_ids=type_ids.unsqueeze(0))[0][0]

        assert predictions.shape == (16, 39)
