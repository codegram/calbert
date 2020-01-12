import logging
import argparse
from pathlib import Path
from typing import Collection

from tokenizers import pre_tokenizers, decoders, trainers, Tokenizer
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from typing import Optional, List, Union

log = logging.getLogger(__name__)


class CalbertTokenizer(BaseTokenizer):
    """ SentencePiece BPE Tokenizer for ALBERT
    Represents the BPE algorithm, with the pretokenization used by SentencePiece
    """

    @classmethod
    def from_dir(cls, tokenizer_dir: Path):
        return CalbertTokenizer(
            str(next(tokenizer_dir.glob("*-vocab.json")).absolute()),
            str(next(tokenizer_dir.glob("*-merges.txt")).absolute()),
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
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>"],
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        show_progress: bool = True,
    ):
        """ Train the model using the given files """

        trainer = trainers.BpeTrainer.new(
            vocab_size=vocab_size,
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
    def sep_token_id(self):
        return self.token_to_id("[SEP]")

    @property
    def cls_token_id(self):
        return self.token_to_id("[CLS]")

    @property
    def mask_token_id(self):
        return self.token_to_id("[MASK]")

    @property
    def unk_token_id(self):
        return self.token_to_id("<unk>")

    @property
    def pad_token_id(self):
        return self.token_to_id("<pad>")

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

    def process(self, sequence, max_seq_len: int = 0):
        enc = self.encode(sequence)
        if max_seq_len > 0:
            enc.pad(max_seq_len, pad_token="<pad>", pad_id=self.pad_token_id)
            enc.truncate(max_seq_len)
        return enc


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer on some raw text")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def train(args, cfg) -> (Tokenizer, int):
    log.info(f"Training tokenizer: {args}")

    tokenizer = CalbertTokenizer()

    tokenizer.train(
        [str(args.input_file.absolute())],
        vocab_size=cfg.vocab.size,
        min_frequency=cfg.vocab.min_frequency,
        special_tokens=["<unk>", "<pad>", "[MASK]", "[SEP]", "[CLS]"],
    )

    vocab_size = tokenizer._tokenizer.get_vocab_size()
    tokenizer.save(str(args.out_dir.absolute()), f"ca.bpe.{vocab_size}")

    log.info(
        f"Saved tokenizer as {args.out_dir.absolute()}/ca.bpe.{vocab_size}[-vocab.json|-merges.txt]"
    )

    return tokenizer, vocab_size
