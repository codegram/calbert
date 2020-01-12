import logging
import argparse
from pathlib import Path
from typing import Collection

from tokenizers import (
    SentencePieceBPETokenizer,
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

    tokenizer = SentencePieceBPETokenizer()

    tokenizer.train(
        [str(args.input_file.absolute())],
        vocab_size=cfg.vocab.size,
        min_frequency=cfg.vocab.min_frequency,
        special_tokens=["<unk>", "<pad>", "[CLS]", "[SEP]", "[MASK]"],
    )

    vocab_size = tokenizer._tokenizer.get_vocab_size()
    tokenizer.save(str(args.out_dir.absolute()), f"ca.bpe.{vocab_size}")

    log.info(
        f"Saved tokenizer as {args.out_dir.absolute()}/ca.bpe.{vocab_size}-vocab.txt"
    )

    return tokenizer, vocab_size


class CalbertTokenizer:
    def __init__(self, vocab_path: Path, merges_path: Path):
        self.tokenizer = SentencePieceBPETokenizer(vocab_path, merges_path)
        self.sep_token_id = self.tokenizer.token_to_id("[SEP]")
        self.cls_token_id = self.tokenizer.token_to_id("[CLS]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.unk_token_id = self.tokenizer.token_to_id("<unk>")

    def __len__(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()

    def encode(self, text: str, seq_len: int = 0) -> Collection[int]:
        tok = self.tokenizer.encode(text)
        if seq_len > 0:
            tok.truncate(seq_len)
        n = len(tok.ids)
        return (
            [self.cls_token_id]
            + tok.ids
            + [self.sep_token_id]
            + ([self.pad_token_id] * max([0, seq_len - n]))
        )

    def decode(self, ids: Collection[int]):
        return [self.decode_one(id) for id in ids]

    def decode_one(self, id: int):
        return self.tokenizer.id_to_token(id)

    def is_special_token(self, token_id: int) -> bool:
        return token_id in [
            self.sep_token_id,
            self.cls_token_id,
            self.mask_token_id,
            self.unk_token_id,
            self.pad_token_id,
        ]

    def get_special_tokens_mask(self, ids):
        return [1 if self.is_special_token(t) else 0 for t in ids]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
            Build model inputs from a sequence or a pair of sequence for sequence classification tasks
            by concatenating and adding special tokens.
            An ALBERT sequence has the following format:
                single sequence: [CLS] X [SEP]
                pair of sequences: [CLS] A [SEP] B [SEP]
            """
        sep_token = [self.sep_token_id]
        cls_token = [self.cls_token_id]
        if token_ids_1 is None:
            return cls_token + token_ids_0 + sep_token
        return cls_token + token_ids_0 + sep_token + token_ids_1 + sep_token
