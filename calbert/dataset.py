import argparse
import logging
import pickle
import io
from pathlib import Path

import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange

from .tokenizer import CalbertTokenizer
from .utils import path_to_str

log = logging.getLogger(__name__)

TRAIN_LMDB_MAP_SIZE = 5.5e10  # 55 GB
VALID_LMDB_MAP_SIZE = 4e9  # 4 GB


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a pickled dataset from raw text."
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def _process_file(
    tokenizer: CalbertTokenizer,
    file_path: Path,
    out_filename: Path,
    max_seq_length: int,
    size: int,
):
    env = lmdb.open(
        path_to_str(out_filename), map_size=size, writemap=True, map_async=True,
    )

    examples = []
    path = path_to_str(file_path)
    num_lines = sum(1 for line in open(path, "r"))
    input_text = tqdm(open(path, encoding="utf-8"), desc="Tokenizing", total=num_lines)
    with env.begin(write=True) as txn:
        for idx, line in enumerate(input_text):
            encoded = tokenizer.process(line.strip(), max_seq_len=max_seq_length)
            example = torch.stack(
                [
                    torch.tensor(encoded.ids),
                    torch.tensor(encoded.attention_mask),
                    torch.tensor(encoded.type_ids),
                ]
            )
            txn.put("{}".format(idx).encode("ascii"), pickle.dumps(example))

    log.info("Saving features into cached file %s", str(out_filename))


def process(args, cfg):
    log.info(f"Creating dataset: {args}")

    tokenizer = CalbertTokenizer.from_dir(args.tokenizer_dir)

    _process_file(
        tokenizer,
        args.train_file,
        args.out_dir / f"train_{cfg.training.max_seq_length}.lmdb",
        cfg.training.max_seq_length,
        TRAIN_LMDB_MAP_SIZE,
    )
    _process_file(
        tokenizer,
        args.valid_file,
        args.out_dir / f"valid_{cfg.training.max_seq_length}.lmdb",
        cfg.training.max_seq_length,
        VALID_LMDB_MAP_SIZE,
    )

    log.info(f"Wrote dataset at {str(args.out_dir)}/train|valid.lmdb")


class CalbertDataset(Dataset):
    def __init__(self, file_path):
        super(CalbertDataset, self).__init__()

        self.env = lmdb.open(
            path_to_str(file_path),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]

    def __getitem__(self, index):
        with self.env.begin(write=False, buffers=True) as txn:
            tokens, ids = pickle.load(
                io.StringIO(txn.get("{}".format(index).encode("ascii")))
            )
        return tokens, ids

    def __len__(self):
        return self.length
