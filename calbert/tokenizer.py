import logging
import argparse
from pathlib import Path

from tokenizers import (
    BertWordPieceTokenizer,
    Tokenizer,
)

log = logging.getLogger(__name__)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer on some raw text")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def train(args, cfg) -> (Tokenizer, int):
    log.info(f"Training tokenizer: {args}")
    log.info(cfg.pretty())

    tokenizer = BertWordPieceTokenizer(
        strip_accents=False, lowercase=cfg.vocab.lowercase
    )

    tokenizer.train([str(args.input_file.absolute())], vocab_size=cfg.vocab.size)

    vocab_size = tokenizer._tokenizer.get_vocab_size()
    tokenizer.save(str(args.out_dir.absolute()), f"ca.bert.{vocab_size}")

    log.info(
        f"Saved tokenizer as {args.out_dir.absolute()}/ca.bert.{vocab_size}-vocab.txt"
    )

    return tokenizer, vocab_size
