from pathlib import Path
from typing import Tuple, List
import argparse
import logging
from functools import partial

import torch
import torch.nn as nn
from fastai2.basics import Learner, Transform, rank_distrib, random, noop, to_device
from fastai2.callback import progress, schedule, fp16
from fastai2.callback.all import SaveModelCallback, ReduceLROnPlateau
from fastai2.metrics import accuracy, Perplexity
from fastai2.data.core import TfmdDL, DataLoaders
from fastai2.optimizer import Lamb
from fastai2.distributed import setup_distrib, DistributedDL, DistributedTrainer
from torch.nn.parallel import DistributedDataParallel
import wandb

from transformers import (
    AlbertConfig,
    AlbertForMaskedLM,
)

from calbert.dataset import CalbertDataset
from calbert.tokenizer import CalbertTokenizer
from calbert.utils import normalize_path
from calbert.reporting import WandbReporter

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
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="The folder where ca.bpe.VOCABSIZE-vocab.json and ca.bpe.VOCABSIZE-merges.txt are",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Where the {train|vaild}.VOCABSIZE.*.npy files are.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--wandb-dir",
        default="wandb",
        type=Path,
        help="The output directory where the Wandb logs will be written to.",
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
        "--subset", default=1.0, type=float, help="Percentage of dataset to use",
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


def mask_tokens(
    inputs: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    tok: CalbertTokenizer,
    ignore_index: int,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, cfg.training.masked_lm_prob)
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = ignore_index  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    inputs[indices_replaced] = tok.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )

    le = len(tok)
    random_words = torch.randint(le, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


class TrainableAlbert(AlbertForMaskedLM):
    def __init__(self, config: AlbertConfig):
        super(TrainableAlbert, self).__init__(config)

    def forward(self, masked_inputs, labels, attention_masks, token_type_ids):
        return super().forward(
            masked_inputs,
            masked_lm_labels=labels,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )[0]


class MaskedSentencePair(Tuple):
    def show(self, ctx=None, **kwargs):
        masked_ids, labels, attention_masks, token_type_ids = self
        return ctx["tokenizer"].decode(masked_ids)


class Mask(Transform):
    def __init__(self, tok: CalbertTokenizer, cfg):
        self.tok = tok
        self.cfg = cfg

    def encodes(self, example) -> MaskedSentencePair:
        ids, special_tokens_mask, attention_masks, token_type_ids = example
        masked_ids, labels = mask_tokens(
            ids,
            special_tokens_mask,
            tok=self.tok,
            cfg=self.cfg,
            ignore_index=IGNORE_INDEX,  # PyTorch CrossEntropyLoss defaults to ignoring -100
        )
        return MaskedSentencePair(
            (masked_ids, labels, attention_masks, token_type_ids, False)
        )


def albert_config(cfg, args) -> AlbertConfig:
    wandb.config.update(cfg.model, allow_val_change=True)
    wandb.config.update(cfg.training, allow_val_change=True)
    wandb.config.update(args)

    model_name = (
        f"calbert-{cfg.model.name}-{'uncased' if cfg.vocab.lowercase else 'cased'}"
    )

    wandb.config["model_name"] = model_name

    return AlbertConfig(vocab_size=cfg.vocab.max_size, **dict(cfg.model))


def initialize_model(cfg, args, tokenizer: CalbertTokenizer) -> TrainableAlbert:
    config = albert_config(cfg, args)
    model = TrainableAlbert(config)

    # Hack until this is released: https://github.com/huggingface/transformers/commit/100e3b6f2133074bf746a78eb4c3a0ad3e939b5f#diff-961191c6a9886609f732e878f0a1fa30
    # Otherwise the resizing doesn't apply for some layers
    # model.predictions.decoder.bias = model.predictions.bias

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
    args, cfg, device: torch.device, tokenizer: CalbertTokenizer, subset=1.0
) -> DataLoaders:
    train_ds = CalbertDataset(
        args.dataset_dir,
        "train",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        subset=subset,
    )
    valid_ds = CalbertDataset(
        args.dataset_dir,
        "valid",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        subset=subset,
    )

    train_dl = TfmdDL(
        train_ds,
        bs=args.train_batch_size,
        after_item=[Mask(tok=tokenizer, cfg=cfg)],
        after_batch=to_device,
        shuffle=True,
        num_workers=0,
    )
    valid_dl = TfmdDL(
        valid_ds,
        bs=args.eval_batch_size,
        after_item=[Mask(tok=tokenizer, cfg=cfg)],
        after_batch=to_device,
        num_workers=0,
    )

    return DataLoaders(train_dl, valid_dl, device=device)


def get_learner(
    cfg,
    model_path: Path,
    dataloaders: DataLoaders,
    model: TrainableAlbert,
    tokenizer: CalbertTokenizer,
) -> Learner:
    learner = Learner(
        dataloaders,
        model,
        loss_func=lambda loss, _: loss,
        opt_func=partial(Lamb, lr=0.1, wd=cfg.training.weight_decay),
        metrics=[Perplexity()],
    )
    learner.model_path = model_path
    learner.add_cbs(
        [
            WandbReporter(
                tokenizer=tokenizer,
                log="all",
                log_preds=True,
                log_examples_html=True,
                model_class=TrainableAlbert,
                ignore_index=IGNORE_INDEX,
            ),
            SaveModelCallback(every_epoch=True),
            ReduceLROnPlateau(monitor="valid_loss", min_delta=0.1, patience=2),
        ]
    )
    return learner


def train(args, cfg) -> Learner:
    args.distributed = args.local_rank != -1
    args.main_process = args.local_rank in [-1, 0]

    run_tags = [
        cfg.model.name,
        "uncased" if cfg.vocab.lowercase else "cased",
        f"sl{cfg.training.max_seq_length}",
        "lamb",
        "sentence-pairs",
    ]

    model_name = "-".join(run_tags[0:3])

    args.tokenizer_dir = normalize_path(args.tokenizer_dir)
    args.dataset_dir = normalize_path(args.dataset_dir)
    args.wandb_dir = normalize_path(args.wandb_dir)
    args.out_dir = normalize_path(args.out_dir)

    args.wandb_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project="calbert", name=model_name, tags=run_tags, dir=str(args.wandb_dir)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CalbertTokenizer.from_dir(
        args.tokenizer_dir, max_seq_length=cfg.training.max_seq_length,
    )
    model = initialize_model(cfg, args, tokenizer=tokenizer)

    device = get_device(args)

    dls = dataloaders(args, cfg, device=device, tokenizer=tokenizer, subset=args.subset)

    learn = get_learner(
        cfg, model_path=args.out_dir, dataloaders=dls, model=model, tokenizer=tokenizer
    )

    if args.fp16:
        learn = learn.to_fp16()

    if args.distributed:
        setup_distrib(args.local_rank)
        learn.add_cb(FixedDistributedTrainer(args.local_rank))
        if rank_distrib() > 0:
            learn.remove_cb(learn.progress)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.main_process else logging.WARN,
    )
    if args.main_process:
        log.info(f"Pretraining ALBERT: {args}")
        log.info(f"Configuration: {cfg.pretty()}")
        log.info(
            f"Sentence pairs: {len(dls.train_ds)} ({args.subset * 100}% of the dataset)"
        )
        log.warning(
            "GPUs: %s, distributed training: %s, 16-bits training: %s",
            torch.cuda.device_count(),
            args.distributed,
            args.fp16,
        )

    learn.fit_one_cycle(cfg.training.epochs, lr_max=cfg.training.learning_rate)

    learn.model.eval()

    learn.save("final")

    return learn
