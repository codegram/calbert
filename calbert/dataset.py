from pathlib import Path
from typing import Tuple
import itertools
import re

import torch
from fastai2.basics import Transform, to_device
from fastai2.text.data import TensorText
from fastai2.data.core import TfmdDL, DataLoaders, Datasets
from torch.utils.data import Dataset, TensorDataset, IterableDataset, DataLoader
from tqdm import tqdm, trange
from transformers import AlbertTokenizer
from collections import namedtuple

SentencePair = namedtuple("SentencePair", ["first", "second"])

IGNORE_INDEX = -100  # Pytorch CrossEntropyLoss defaults to ignoring -100


def chunk(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # chunk('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    yield from itertools.zip_longest(*[iter(iterable)] * n, fillvalue=fillvalue)


punctuation = re.compile(r"[\.!\?]+")


def sentence_pairs(filename, min_length=8, max_items=None):
    with open(filename, encoding="utf-8") as f:
        counter = 0
        for line in f:
            sentences = [
                s.strip()
                for s in punctuation.split(line)
                if len(s) >= min_length and " " in s
            ]
            for a, b in itertools.zip_longest(sentences[:-1], sentences[1:]):
                if (not max_items) or (max_items and counter < max_items):
                    counter += 1
                    yield SentencePair(a + ".", b + ".")


class Tokenize(Transform):
    order = 17

    def __init__(self, tokenizer: AlbertTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def encodes(self, inp: SentencePair) -> TensorText:
        tokenized = self.tokenizer.batch_encode_plus(
            [inp],
            max_length=self.max_seq_len,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        return TensorText(
            torch.stack(
                [
                    tokenized["input_ids"].squeeze(),
                    tokenized["attention_mask"].squeeze(),
                    tokenized["token_type_ids"].squeeze(),
                ]
            )
        )

    def decodes(self, encoded: TensorText):
        enc = encoded if encoded.ndim == 1 else encoded[0]
        return self.tokenizer.decode(
            enc, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


class CalbertDataset(IterableDataset):
    def __init__(self, dataset_path: Path, max_items=None):
        super(CalbertDataset, self).__init__()
        self.path = dataset_path
        self.max_items = max_items

    def __iter__(self):
        return sentence_pairs(self.path, max_items=self.max_items)


def mask_tokens(
    inputs: torch.Tensor, tok: AlbertTokenizer, ignore_index: int, probability: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    special_tokens_mask = (
        (inputs == tok.cls_token_id)
        | (inputs == tok.pad_token_id)
        | (inputs == tok.sep_token_id)
    )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
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


class Mask(Transform):
    order = 18

    def __init__(self, tok: AlbertTokenizer, probability: float):
        self.tok = tok
        self.probability = probability

    def encodes(self, example: TensorText):
        ids, attention_masks, token_type_ids = example
        masked_ids, labels = mask_tokens(
            ids,
            tok=self.tok,
            probability=self.probability,
            ignore_index=IGNORE_INDEX,  # PyTorch CrossEntropyLoss defaults to ignoring -100
        )
        return torch.stack([masked_ids, labels, attention_masks, token_type_ids])


@Transform
def ignore(x):
    return 0


def dataloaders(
    args, cfg, tokenizer: AlbertTokenizer, tds: CalbertDataset, vds: CalbertDataset
) -> DataLoaders:
    tfms = [
        Tokenize(tokenizer, max_seq_len=cfg.training.max_seq_length),
        Mask(tok=tokenizer, probability=cfg.training.masked_lm_prob),
    ]

    train_ds = Datasets(tds, tfms=[tfms, [ignore]])
    valid_ds = Datasets(vds, tfms=[tfms, [ignore]])

    return DataLoaders(
        TfmdDL(
            train_ds,
            batch_size=args.train_batch_size,
            num_workers=0,
            after_batch=to_device,
        ),
        TfmdDL(
            valid_ds,
            batch_size=args.eval_batch_size,
            num_workers=0,
            after_batch=to_device,
        ),
    )
