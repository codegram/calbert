from pathlib import Path
from collections import ChainMap
from typing import Tuple, List
import argparse
import logging
from functools import partial

import deepkit
import torch
import torch.nn as nn
from fastai2.basics import Learner, Transform, rank_distrib, random, noop, to_device
from fastai2.callback import progress, schedule, fp16
from fastai2.callback.all import SaveModelCallback, ReduceLROnPlateau
from fastai2.metrics import accuracy, Perplexity
from fastai2.data.core import TfmdDL, DataLoaders, Datasets
from fastai2.text.data import TensorText
from fastai2.optimizer import Lamb
from fastai2.distributed import setup_distrib, DistributedDL, DistributedTrainer
from torch.nn.parallel import DistributedDataParallel

from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
)
from transformers.modeling_albert import AlbertMLMHead

from calbert.reporting import DeepkitCallback
from calbert.dataset import CalbertDataset, Tokenize, dataloaders as build_dataloaders
from calbert.tokenizer import AlbertTokenizer, load as load_tokenizer
from calbert.utils import normalize_path

log = logging.getLogger(__name__)

IGNORE_INDEX = -100  # Pytorch CrossEntropyLoss defaults to ignoring -100


class FixedDistributedTrainer(DistributedTrainer):
    def begin_fit(self):
        self.learn.model = DistributedDataParallel(
            self.model,
            device_ids=[self.cuda_id],
            output_device=self.cuda_id,
            find_unused_parameters=True,
        )
        self.old_dls = list(self.dls)
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
        if rank_distrib() > 0:
            self.learn.logger = noop


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ALBERT")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="The path to the sentencepiece *model* (ca.{uncased|cased}.VOCABSIZE.model)",
    )
    parser.add_argument(
        "--train-path", required=True, type=Path, help="Where the train.txt file lives",
    )
    parser.add_argument(
        "--valid-path", required=True, type=Path, help="Where the valid.txt file lives",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        type=Path,
        help="The optional output directory where to save the model in HuggingFace format",
    )

    parser.add_argument(
        "--train-batch-size",
        default=128,
        type=int,
        help="Batch size across all GPUs/CPUs for training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        default=128,
        type=int,
        help="Batch size across all GPUs/CPUs for evaluation.",
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="Number of epochs to train",
    )

    parser.add_argument(
        "--max-items",
        default=None,
        type=int,
        help="Number of sentence pairs to use (defaults to all)",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Node number -- set by Pytorch in distributed training",
    )
    return parser


class CalbertForMaskedLM(AlbertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input):
        input_ids, masked_lm_labels, attention_mask, token_type_ids = input.permute(
            1, 0, 2
        )

        position_ids = None
        head_mask = None
        inputs_embeds = None

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        outputs = (prediction_scores,) + outputs[
            2:
        ]  # Add hidden states and attention if they are here
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.reshape(-1),
            )
            outputs = (masked_lm_loss,) + outputs

        return outputs[0]  # return only loss


def albert_config(cfg, args) -> AlbertConfig:
    model_name = (
        f"calbert-{cfg.model.name}-{'uncased' if cfg.vocab.lowercase else 'cased'}"
    )

    return AlbertConfig(vocab_size=cfg.vocab.max_size, **dict(cfg.model))


def initialize_model(cfg, args, tokenizer: AlbertTokenizer) -> CalbertForMaskedLM:
    config = albert_config(cfg, args)
    model = CalbertForMaskedLM(config)

    model_to_resize = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))
    to_device(model)
    return model


def get_device(args) -> torch.device:
    if args.local_rank == -1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device("cuda", args.local_rank)


def dataloaders(
    args, cfg, device: torch.device, tokenizer: AlbertTokenizer, max_items=None
) -> DataLoaders:
    train_ds = CalbertDataset(args.train_path, max_items=max_items,)
    valid_ds = CalbertDataset(args.valid_path, max_items=max_items,)

    return build_dataloaders(args, cfg, tokenizer, train_ds, valid_ds)


def get_learner(
    args,
    cfg,
    dataloaders: DataLoaders,
    model: CalbertForMaskedLM,
    tokenizer: AlbertTokenizer,
) -> Learner:
    learner = Learner(
        dataloaders,
        model,
        loss_func=lambda loss, _: loss,
        opt_func=partial(Lamb, lr=0.1, wd=cfg.training.weight_decay),
        metrics=[Perplexity()],
    )
    cbs = []
    if args.distributed:
        setup_distrib(args.local_rank)
        cbs.append(FixedDistributedTrainer(args.local_rank))
    cbs.extend([DeepkitCallback(args, cfg)])
    learner.add_cbs(cbs)
    if args.distributed and rank_distrib() > 0:
        learner.remove_cb(learner.progress)
    return learner


def set_config(experiment, key, val):
    if key not in ["_resolver_cache", "content", "flags"] and val is not None:
        if isinstance(val, int) or isinstance(val, float):
            experiment.set_config(key, val)
        else:
            experiment.set_config(key, str(val))


def train(args, cfg, test_mode=False) -> Learner:
    experiment = deepkit.experiment()
    if test_mode:
        experiment.set_title("Unit test")
        experiment.set_info("test", True)

    for key, val in vars(args).items():
        set_config(experiment, key, val)

    for key, val in dict(cfg.vocab).items():
        set_config(experiment, f"vocab.{key}", val)
    for key, val in dict(cfg.training).items():
        set_config(experiment, f"training.{key}", val)
    for key, val in dict(cfg.model).items():
        set_config(experiment, f"model.{key}", val)

    args.experiment = experiment

    args.distributed = args.local_rank != -1
    args.main_process = args.local_rank in [-1, 0]

    run_tags = [
        cfg.model.name,
        "uncased" if cfg.vocab.lowercase else "cased",
        f"sl{cfg.training.max_seq_length}",
    ]

    model_name = "-".join(run_tags[0:3])

    args.tokenizer_path = normalize_path(args.tokenizer_path)
    args.train_path = normalize_path(args.train_path)
    args.valid_path = normalize_path(args.valid_path)

    tokenizer = load_tokenizer(cfg, args.tokenizer_path)
    model = initialize_model(cfg, args, tokenizer=tokenizer)

    device = get_device(args)

    dls = dataloaders(
        args, cfg, device=device, tokenizer=tokenizer, max_items=args.max_items
    )

    learn = get_learner(args, cfg, dataloaders=dls, model=model, tokenizer=tokenizer,)

    if args.fp16:
        learn = learn.to_fp16()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.main_process else logging.WARN,
    )
    if args.main_process:
        log.info(f"Pretraining ALBERT: {args}")
        log.info(f"Configuration: {cfg.pretty()}")
        if args.max_items:
            log.info(f"Sentence pairs limited to {args.max_items}")
        else:
            log.info(f"Processing all sentence pairs")
        log.warning(
            "GPUs: %s, distributed training: %s, 16-bits training: %s",
            torch.cuda.device_count(),
            args.distributed,
            args.fp16,
        )

    learn.fit_one_cycle(args.epochs, lr_max=cfg.training.learning_rate)

    learn.model.eval()

    if args.export_path:
        args.export_path = normalize_path(args.export_path)
        args.export_path.mkdir(parents=True, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.__class__ = AlbertForMaskedLM
        torch.save(model_to_save.state_dict(), args.export_path / "pytorch_model.bin")
        model_to_save.config.to_json_file(args.export_path / "config.json")
        tokenizer.save_pretrained(args.export_path)
        for file in args.export_path.glob("*"):
            args.experiment.add_output_file(str(file))

    return learn
