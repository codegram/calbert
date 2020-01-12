import argparse
import logging
import pickle
from pathlib import Path

import torch
from tqdm import tqdm, trange

from .tokenizer import CalbertTokenizer

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


def _process_file(tokenizer, file_path, out_filename, max_seq_length):
    examples = []
    path = str(file_path.absolute())
    num_lines = sum(1 for line in open(path, "r"))
    input_text = tqdm(open(path, encoding="utf-8"), desc="Tokenizing", total=num_lines)
    for line in input_text:
        encoded = tokenizer.process(line.strip(), max_seq_len=max_seq_length)
        examples.append(
            torch.stack(
                [
                    torch.tensor(encoded.ids),
                    torch.tensor(encoded.attention_mask),
                    torch.tensor(encoded.type_ids),
                ]
            )
        )

    log.info("Saving features into cached file %s", str(out_filename))
    with open(str(out_filename.absolute()), "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process(args, cfg):
    log.info(f"Creating dataset: {args}")

    tokenizer = CalbertTokenizer.from_dir(args.tokenizer_dir)

    _process_file(
        tokenizer,
        args.train_file,
        args.out_dir / "train.pkl",
        cfg.training.max_seq_length,
    )
    _process_file(
        tokenizer,
        args.valid_file,
        args.out_dir / "valid.pkl",
        cfg.training.max_seq_length,
    )

    log.info(f"Wrote dataset at {str(args.out_dir)}/train|valid.pkl")
