import argparse
import logging
import itertools
import pickle
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from .tokenizer import CalbertTokenizer
from .utils import path_to_str

log = logging.getLogger(__name__)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a pickled dataset from raw text."
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def chunk(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # chunk('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    yield from itertools.zip_longest(*[iter(iterable)] * n, fillvalue=fillvalue)


def _process_file(
    tokenizer: CalbertTokenizer,
    input_file: Path,
    out_dir: Path,
    split: str,
    max_seq_length: int,
    max_vocab_size: int,
    minibatch_size: int,
):
    assert minibatch_size > 0
    num_lines = sum(1 for line in open(path_to_str(input_file), "r"))
    input_text = tqdm(
        map(
            lambda minibatch: filter(lambda line: line is not None, minibatch),
            chunk(
                open(path_to_str(input_file), encoding="utf-8"), minibatch_size, None,
            ),
        ),
        desc="Tokenizing",
        total=int(num_lines / minibatch_size),
    )

    with open(
        dataset_element(
            out_dir, split, max_seq_length, max_vocab_size, "count", npy=False
        ),
        "w",
    ) as f:

        f.write(str(num_lines))

    ids_type = dataset_type("ids", max_vocab_size)
    ids_np = np.memmap(
        dataset_element(out_dir, split, max_seq_length, max_vocab_size, "ids"),
        dtype=ids_type,
        mode="w+",
        shape=(num_lines, max_seq_length),
    )
    special_tokens_mask_type = dataset_type("special_tokens_mask", max_vocab_size)
    special_tokens_mask_np = np.memmap(
        dataset_element(
            out_dir, split, max_seq_length, max_vocab_size, "special_tokens_mask"
        ),
        dtype=special_tokens_mask_type,
        mode="w+",
        shape=(num_lines, max_seq_length),
    )
    attention_mask_type = dataset_type("attention_mask", max_vocab_size)
    attention_mask_np = np.memmap(
        dataset_element(
            out_dir, split, max_seq_length, max_vocab_size, "attention_mask"
        ),
        dtype=attention_mask_type,
        mode="w+",
        shape=(num_lines, max_seq_length),
    )
    type_ids_type = dataset_type("type_ids", max_vocab_size)
    type_ids_np = np.memmap(
        dataset_element(out_dir, split, max_seq_length, max_vocab_size, "type_ids"),
        dtype=type_ids_type,
        mode="w+",
        shape=(num_lines, max_seq_length),
    )
    for minibatch_idx, lines in enumerate(input_text):
        start_idx = minibatch_idx * minibatch_size
        end_idx = start_idx + minibatch_size
        encodings = tokenizer.encode_batch([line.strip() for line in lines])
        ids_np[start_idx:end_idx, :] = np.array(
            [e.ids for e in encodings], dtype=ids_type
        )
        special_tokens_mask_np[start_idx:end_idx, :] = np.array(
            [e.special_tokens_mask for e in encodings], dtype=special_tokens_mask_type
        )
        attention_mask_np[start_idx:end_idx, :] = np.array(
            [e.attention_mask for e in encodings], dtype=attention_mask_type
        )
        type_ids_np[start_idx:end_idx, :] = np.array(
            [e.type_ids for e in encodings], dtype=type_ids_type
        )

    del ids_np
    del special_tokens_mask_np
    del attention_mask_np
    del type_ids_np

    log.info(
        "Saving features into files %s/[SPLIT].v[MAXVOCABSIZE].sl[MAXSEQLEN].PART.npy",
        str(out_dir),
    )


def process(args, cfg):
    log.info(f"Creating dataset: {args}")

    tokenizer = CalbertTokenizer.from_dir(
        args.tokenizer_dir, cfg.training.max_seq_length
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _process_file(
        tokenizer,
        args.train_file,
        args.out_dir,
        split="train",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        minibatch_size=cfg.data.processing_minibatch_size,
    )
    _process_file(
        tokenizer,
        args.valid_file,
        args.out_dir,
        split="valid",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        minibatch_size=cfg.data.processing_minibatch_size,
    )

    log.info(
        f"Wrote dataset at {str(args.out_dir)}/train|valid.v{cfg.vocab.max_size}.sl{cfg.training.max_seq_length}.*.npy"
    )


def dataset_element(
    dataset_dir: Path,
    split: str,
    max_seq_length: int,
    max_vocab_size: int,
    element: str,
    npy=True,
) -> str:
    return f"{path_to_str(dataset_dir)}/{split}.v{max_vocab_size}.sl{max_seq_length}.{element}{'.npy' if npy else ''}"


def dataset_type(element, max_vocab_size):
    if element == "ids":
        return np.int16 if max_vocab_size < 32000 else np.int32
    else:
        return np.bool


class CalbertDataset(Dataset):
    def __init__(
        self, dataset_dir: Path, split: str, max_seq_length: int, max_vocab_size: int
    ):
        super(CalbertDataset, self).__init__()

        self.length = int(
            open(
                dataset_element(
                    dataset_dir,
                    split,
                    max_seq_length,
                    max_vocab_size,
                    "count",
                    npy=False,
                )
            )
            .read()
            .strip()
        )

        self.ids = np.memmap(
            dataset_element(dataset_dir, split, max_seq_length, max_vocab_size, "ids"),
            dtype=dataset_type("ids", max_vocab_size),
            mode="r",
            shape=(self.length, max_seq_length),
        )
        self.special_tokens_mask = np.memmap(
            dataset_element(
                dataset_dir,
                split,
                max_seq_length,
                max_vocab_size,
                "special_tokens_mask",
            ),
            dtype=dataset_type("special_tokens_mask", max_vocab_size),
            mode="r",
            shape=(self.length, max_seq_length),
        )
        self.attention_mask = np.memmap(
            dataset_element(
                dataset_dir, split, max_seq_length, max_vocab_size, "attention_mask"
            ),
            dtype=dataset_type("attention_mask", max_vocab_size),
            mode="r",
            shape=(self.length, max_seq_length),
        )
        self.type_ids = np.memmap(
            dataset_element(
                dataset_dir, split, max_seq_length, max_vocab_size, "type_ids"
            ),
            dtype=dataset_type("type_ids", max_vocab_size),
            mode="r",
            shape=(self.length, max_seq_length),
        )

    def __getitem__(self, index):
        return torch.tensor(
            np.stack(
                [
                    self.ids[index],
                    self.special_tokens_mask[index],
                    self.attention_mask[index],
                    self.type_ids[index],
                ]
            )
        ).long()

    def __len__(self):
        return self.length
