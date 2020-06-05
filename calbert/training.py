from pathlib import Path
from collections import ChainMap
from typing import Tuple, List
import argparse
import logging
from functools import partial

from fastprogress import fastprogress
import deepkit
import torch
import torch.nn as nn
from fastai2.basics import (
    Learner,
    Transform,
    random,
    noop,
    to_device,
    default_device,
)
from fastai2.callback import progress, schedule, fp16
from fastai2.callback.all import SaveModelCallback, ReduceLROnPlateau
from fastai2.distributed import (
    rank_distrib,
    DistributedTrainer,
    distrib_ctx,
    num_distrib,
)
from fastai2.metrics import accuracy, Perplexity
from fastai2.data.core import TfmdDL, DataLoaders, Datasets
from fastai2.text.data import TensorText
from fastai2.optimizer import Lamb

from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
)
from transformers.modeling_albert import AlbertMLMHead

from calbert.reporting import DeepkitCallback
from calbert.dataset import CalbertDataset, Tokenize, dataloaders as build_dataloaders
from calbert.model import CalbertForMaskedLM
from calbert.tokenizer import AlbertTokenizer, load as load_tokenizer
from calbert.utils import normalize_path

fastprogress.MAX_COLS = 80

log = logging.getLogger(__name__)

IGNORE_INDEX = -100  # Pytorch CrossEntropyLoss defaults to ignoring -100


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
        "--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision",
    )
    parser.add_argument(
        "--deepkit",
        action="store_true",
        help="Whether to log metrics and insights to Deepkit",
    )
    parser.add_argument(
        "--gpu", default=None, type=int,
    )
    return parser


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
    return to_device(model, default_device())


def dataloaders(args, cfg, tokenizer: AlbertTokenizer, max_items=None) -> DataLoaders:
    train_ds = CalbertDataset(args.train_path, max_items=max_items,)
    valid_ds = CalbertDataset(args.valid_path, max_items=max_items,)

    return build_dataloaders(args, cfg, tokenizer, train_ds, valid_ds)


def get_learner(
    args,
    cfg,
    dataloaders: DataLoaders,
    model: CalbertForMaskedLM,
    tokenizer: AlbertTokenizer,
    use_deepkit: False,
) -> Learner:
    learner = Learner(
        dataloaders,
        model,
        loss_func=lambda out, _: out[0],
        opt_func=partial(Lamb, lr=0.1, wd=cfg.training.weight_decay),
        metrics=[Perplexity()],
    )
    cbs = []
    if use_deepkit:
        cbs.extend([DeepkitCallback(args, cfg, tokenizer)])
    learner.add_cbs(cbs)
    return learner


def set_config(experiment, key, val):
    if key not in ["_resolver_cache", "content", "flags"] and val is not None:
        if isinstance(val, int) or isinstance(val, float):
            experiment.set_config(key, val)
        else:
            experiment.set_config(key, str(val))


def train(args, cfg) -> Learner:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if args.gpu is None:
            args.gpu = list(range(n_gpu))[0]
        torch.cuda.set_device(args.gpu)
    else:
        n_gpu = None
        args.gpu = -1

    use_deepkit = args.deepkit and rank_distrib() == 0
    if use_deepkit:
        experiment = deepkit.experiment()

        for key, val in vars(args).items():
            set_config(experiment, key, val)

        for key, val in dict(cfg.vocab).items():
            set_config(experiment, f"vocab.{key}", val)
        for key, val in dict(cfg.training).items():
            set_config(experiment, f"training.{key}", val)
        for key, val in dict(cfg.model).items():
            set_config(experiment, f"model.{key}", val)

        args.experiment = experiment

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

    dls = dataloaders(args, cfg, tokenizer=tokenizer, max_items=args.max_items)
    dls.to(default_device())

    learn = get_learner(args, cfg, dataloaders=dls, model=model, tokenizer=tokenizer, use_deepkit=use_deepkit)

    if args.fp16:
        learn = learn.to_fp16()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    log.info(f"Model device is {model.device}, loader device is {dls[0].device}")

    if rank_distrib() == 0:
        log.info(f"Pretraining ALBERT: {args}")
        log.info(f"Configuration: {cfg.pretty()}")

        if args.max_items:
            log.info(f"Sentence pairs limited to {args.max_items}")
        else:
            log.info("Processing all sentence pairs")
        log.info(
            "GPUs: %s, 16-bits training: %s", torch.cuda.device_count(), args.fp16,
        )

    if num_distrib() > 1:
        DistributedTrainer.fup = True

    with learn.distrib_ctx(
        cuda_id=args.gpu
    ):  # distributed traing requires "-m fastai2.launch"
        log.info(f"Training in distributed data parallel context on GPU {args.gpu}")
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
        if use_deepkit:
            for file in args.export_path.glob("*"):
                args.experiment.add_output_file(str(file))

    return learn
