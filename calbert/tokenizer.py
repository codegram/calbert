import logging
import argparse
from pathlib import Path
from typing import Collection

import torch
from tokenizers import pre_tokenizers, decoders, trainers, Tokenizer, Encoding
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from typing import Optional, List, Union

from .utils import path_to_str

log = logging.getLogger(__name__)


class CalbertTokenizer(BaseTokenizer):
    """ SentencePiece BPE Tokenizer for ALBERT
    Represents the BPE algorithm, with the pretokenization used by SentencePiece
    """

    @classmethod
    def from_dir(cls, tokenizer_dir: Path):
        return CalbertTokenizer(
            path_to_str(next(tokenizer_dir.glob("*-vocab.json"))),
            path_to_str(next(tokenizer_dir.glob("*-merges.txt"))),
        )

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "<unk>",
        replacement: str = "▁",
        add_prefix_space: bool = True,
        dropout: Optional[float] = None,
    ):
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE.from_files(
                    vocab_file, merges_file, dropout=dropout, unk_token=unk_token
                )
            )
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.normalizer = NFKC.new()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace.new(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        tokenizer.decoder = decoders.Metaspace.new(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        tokenizer.post_processor = BertProcessing.new(("[SEP]", 3), ("[CLS]", 4))

        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        max_vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer.new(
            vocab_size=max_vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(trainer, files)

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self):
        return self.token_to_id("<pad>")

    @property
    def mask_token_id(self):
        return self.token_to_id("[MASK]")

    def process(self, sequence: str, pair: Optional[str] = None, max_seq_len: int = 0):
        enc = self.encode(sequence, pair)
        if max_seq_len > 0:
            enc.pad(max_seq_len, pad_token="<pad>", pad_id=self.pad_token_id)
            enc.truncate(max_seq_len)
        return enc


def encoding_to_tensor(e: Encoding) -> torch.Tensor:
    return torch.stack(
        [
            torch.tensor(e.ids),
            torch.tensor(e.special_tokens_mask),
            torch.tensor(e.attention_mask),
            torch.tensor(e.type_ids),
        ]
    )


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer on some raw text")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def train(args, cfg) -> Tokenizer:
    log.info(f"Training tokenizer: {args}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CalbertTokenizer()

    tokenizer.train(
        [path_to_str(args.input_file)],
        max_vocab_size=cfg.vocab.max_size,
        min_frequency=cfg.vocab.min_frequency,
        special_tokens=["<unk>", "<pad>", "[MASK]", "[SEP]", "[CLS]"],
    )

    out_dir = path_to_str(args.out_dir)
    tokenizer.save(out_dir, f"ca.bpe.{len(tokenizer)}")

    log.info(
        f"Saved tokenizer as {out_dir}/ca.bpe.{len(tokenizer)}[-vocab.json|-merges.txt]"
    )

    return tokenizer
