import argparse
import logging
import itertools
import pickle
import io
import re
from pathlib import Path

from diskarray import DiskArray
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from .tokenizer import CalbertTokenizer
from .utils import normalize_path

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


punctuation = re.compile(r"[\.!\?]+")


def sentence_pairs(filename, min_length=8):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            sentences = [
                s.strip()
                for s in punctuation.split(line)
                if len(s) >= min_length and " " in s
            ]
            for a, b in itertools.zip_longest(sentences[:-1], sentences[1:]):
                yield (a + ".", b + ".")


def _load_memmap(
    out_dir, split, kind, max_seq_length, max_vocab_size, dtype, count, minibatch_size
):
    return DiskArray(
        dataset_element(out_dir, split, max_seq_length, max_vocab_size, kind),
        shape=(0, max_seq_length),
        capacity=(minibatch_size, max_seq_length),
        growby=minibatch_size,
        dtype=dtype,
    )


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
    sentence_pair_count = 0
    pairs = tqdm(
        map(
            lambda minibatch: [pair for pair in minibatch if pair is not None],
            chunk(sentence_pairs(input_file), minibatch_size, None),
        ),
        desc="Tokenizing",
        total=int(19557475 / minibatch_size)
    )

    ids_type = dataset_type("ids", max_vocab_size)
    special_tokens_mask_type = dataset_type("special_tokens_mask", max_vocab_size)
    attention_mask_type = dataset_type("attention_mask", max_vocab_size)
    type_ids_type = dataset_type("type_ids", max_vocab_size)

    ids_np = _load_memmap(
        out_dir,
        split,
        "ids",
        max_seq_length,
        max_vocab_size,
        ids_type,
        sentence_pair_count,
        minibatch_size,
    )
    special_tokens_mask_np = _load_memmap(
        out_dir,
        split,
        "special_tokens_mask",
        max_seq_length,
        max_vocab_size,
        special_tokens_mask_type,
        sentence_pair_count,
        minibatch_size,
    )
    attention_mask_np = _load_memmap(
        out_dir,
        split,
        "attention_mask",
        max_seq_length,
        max_vocab_size,
        attention_mask_type,
        sentence_pair_count,
        minibatch_size,
    )
    type_ids_np = _load_memmap(
        out_dir,
        split,
        "type_ids",
        max_seq_length,
        max_vocab_size,
        type_ids_type,
        sentence_pair_count,
        minibatch_size,
    )

    for minibatch_idx, minibatch in enumerate(pairs):
        # Allocate enough disk to write them
        this_minibatch_size = len(minibatch)
        sentence_pair_count += this_minibatch_size

        start_idx = minibatch_idx * minibatch_size
        end_idx = start_idx + this_minibatch_size

        encodings = tokenizer.encode_batch(minibatch)

        ids_np.extend(np.array([e.ids for e in encodings], dtype=ids_type))

        special_tokens_mask_np.extend(
            np.array(
                [e.special_tokens_mask for e in encodings],
                dtype=special_tokens_mask_type,
            )
        )

        attention_mask_np.extend(
            np.array([e.attention_mask for e in encodings], dtype=attention_mask_type)
        )

        type_ids_np.extend(
            np.array([e.type_ids for e in encodings], dtype=type_ids_type)
        )

        # ids_np[start_idx:end_idx, :] = np.array(
        #     [e.ids for e in encodings], dtype=ids_type
        # )
        # special_tokens_mask_np[start_idx:end_idx, :] = np.array(
        #     [e.special_tokens_mask for e in encodings], dtype=special_tokens_mask_type
        # )
        # attention_mask_np[start_idx:end_idx, :] = np.array(
        #     [e.attention_mask for e in encodings], dtype=attention_mask_type
        # )
        # type_ids_np[start_idx:end_idx, :] = np.array(
        #     [e.type_ids for e in encodings], dtype=type_ids_type
        # )

        # ids_np.flush()
        # special_tokens_mask_np.flush()
        # attention_mask_np.flush()
        # type_ids_np.flush()

    # del ids_np
    # del special_tokens_mask_np
    # del attention_mask_np
    # del type_ids_np

    with open(
        dataset_element(
            out_dir, split, max_seq_length, max_vocab_size, "count", npy=False
        ),
        "w",
    ) as f:
        f.write(str(sentence_pair_count))

    log.info(
        "Saving features into files %s/[SPLIT].v[MAXVOCABSIZE].sl[MAXSEQLEN].PART.npy",
        str(out_dir),
    )


def process(args, cfg):
    log.info(f"Creating dataset: {args}")

    train_file = normalize_path(args.train_file)
    valid_file = normalize_path(args.valid_file)
    out_dir = normalize_path(args.out_dir)
    tokenizer_dir = normalize_path(args.tokenizer_dir)

    tokenizer = CalbertTokenizer.from_dir(
        tokenizer_dir, cfg.training.max_seq_length
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    _process_file(
        tokenizer,
        train_file,
        out_dir,
        split="train",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        minibatch_size=cfg.data.processing_minibatch_size,
    )
    _process_file(
        tokenizer,
        valid_file,
        out_dir,
        split="valid",
        max_seq_length=cfg.training.max_seq_length,
        max_vocab_size=cfg.vocab.max_size,
        minibatch_size=cfg.data.processing_minibatch_size,
    )

    log.info(
        f"Wrote dataset at {str(out_dir)}/train|valid.v{cfg.vocab.max_size}.sl{cfg.training.max_seq_length}.*.npy"
    )


def dataset_element(
    dataset_dir: Path,
    split: str,
    max_seq_length: int,
    max_vocab_size: int,
    element: str,
    npy=True,
) -> str:
    return f"{dataset_dir}/{split}.v{max_vocab_size}.sl{max_seq_length}.{element}{'.npy' if npy else ''}"


def dataset_type(element, max_vocab_size):
    if element == "ids":
        return np.int16 if max_vocab_size < 32000 else np.int32
    else:
        return np.bool


class CalbertDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        max_seq_length: int,
        max_vocab_size: int,
        subset: int = 1.0,
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

        dataset_element(dataset_dir, split, max_seq_length, max_vocab_size, "ids"),

        self.effective_length = max(1, int(self.length * subset))

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
        return self.effective_length
