import logging
import argparse
import collections
from pathlib import Path
from typing import Tuple

import torch
from tokenizers import pre_tokenizers, decoders, trainers, Tokenizer, Encoding
from tokenizers.processors import BertProcessing
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from typing import Optional, List, Union

from .utils import normalize_path

log = logging.getLogger(__name__)


CustomEncoding = collections.namedtuple(
    "CustomEncoding",
    ["tokens", "ids", "special_tokens_mask", "attention_mask", "type_ids"],
)


class CalbertTokenizer(BaseTokenizer):
    """ SentencePiece BPE Tokenizer for ALBERT
    Represents the BPE algorithm, with the pretokenization used by SentencePiece
    """

    @classmethod
    def from_dir(cls, tokenizer_dir: Path, max_seq_length: int):
        return CalbertTokenizer(
            max_seq_length=max_seq_length,
            vocab_file=str(next(tokenizer_dir.glob("*-vocab.json"))),
            merges_file=str(next(tokenizer_dir.glob("*-merges.txt"))),
        )

    def __init__(
        self,
        max_seq_length: int,
        lowercase: bool = False,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        unk_token: str = "<unk>",
        replacement: str = "▁",
        add_prefix_space: bool = True,
        dropout: Optional[float] = None,
    ):
        self.max_seq_length = max_seq_length
        if vocab_file is not None and merges_file is not None:
            tokenizer = Tokenizer(
                BPE.from_files(
                    vocab_file, merges_file, dropout=dropout, unk_token=unk_token
                )
            )
        else:
            tokenizer = Tokenizer(BPE.empty())

        tokenizer.normalizer = NFKC.new(lowercase=lowercase)
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace.new(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        tokenizer.decoder = decoders.Metaspace.new(
            replacement=replacement, add_prefix_space=add_prefix_space
        )
        tokenizer.enable_truncation(max_length=max_seq_length)
        tokenizer.enable_padding(pad_token="<pad>", pad_id=1, max_length=max_seq_length)
        tokenizer.post_processor = BertProcessing.new(("[SEP]", 3), ("[CLS]", 4))

        parameters = {
            "model": "SentencePieceBPE",
            "unk_token": unk_token,
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
            "dropout": dropout,
        }

        super().__init__(tokenizer, parameters)

        self.unk_token_id = 0
        self.pad_token_id = 1
        self.mask_token_id = 2
        self.sep_token_id = 3
        self.cls_token_id = 4

    def train(
        self,
        files: Union[str, List[str]],
        max_vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<unk>", "<pad>", "[MASK]", "[SEP]", "[CLS]"],
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

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()

    def _truncate(self, encoding: Encoding) -> CustomEncoding:
        curr_len = len(encoding.tokens)
        max_len = self.max_seq_length

        pair_len = sum(encoding.type_ids) - 1
        first_len = curr_len - pair_len - 2

        longest = 0 if first_len > pair_len else 1

        one_start = 1
        one_end = first_len - 1

        two_start = first_len + 1
        two_end = curr_len - 1

        trimming = "one"
        while (one_end - one_start) + (two_end - two_start) + 3 != max_len:
            if trimming == "one":
                one_end -= 1
            else:
                two_end -= 1

            if (one_end - one_start) > (two_end - two_start):
                trimming = "one"
            else:
                trimming = "two"

        tokens = (
            [encoding.tokens[0]]
            + encoding.tokens[one_start:one_end]
            + [encoding.tokens[-1]]
            + encoding.tokens[two_start:two_end]
            + [encoding.tokens[-1]]
        )
        ids = (
            [encoding.ids[0]]
            + encoding.ids[one_start:one_end]
            + [encoding.ids[-1]]
            + encoding.ids[two_start:two_end]
            + [encoding.ids[-1]]
        )
        special_tokens_mask = (
            [encoding.special_tokens_mask[0]]
            + encoding.special_tokens_mask[one_start:one_end]
            + [encoding.special_tokens_mask[-1]]
            + encoding.special_tokens_mask[two_start:two_end]
            + [encoding.special_tokens_mask[-1]]
        )
        attention_mask = (
            [encoding.attention_mask[0]]
            + encoding.attention_mask[one_start:one_end]
            + [encoding.attention_mask[-1]]
            + encoding.attention_mask[two_start:two_end]
            + [encoding.attention_mask[-1]]
        )
        type_ids = (
            [encoding.type_ids[0]]
            + encoding.type_ids[one_start:one_end]
            + [encoding.type_ids[-1]]
            + encoding.type_ids[two_start:two_end]
            + [encoding.type_ids[-1]]
        )

        return CustomEncoding(
            tokens=tokens,
            ids=ids,
            special_tokens_mask=special_tokens_mask,
            attention_mask=attention_mask,
            type_ids=type_ids,
        )

    def encode(
        self, sentence: str, pair: Optional[str] = None
    ) -> Union[CustomEncoding, Encoding]:
        encoding = super().encode(sentence, pair)
        if pair is None or len(encoding.tokens) == self.max_seq_length:
            return encoding
        else:
            return self._truncate(encoding)

    def encode_batch(
        self, sentences: Union[List[str], List[Tuple[str, str]]]
    ) -> List[Union[Encoding, CustomEncoding]]:
        out = []
        for encoding in super().encode_batch(sentences):
            if len(encoding.tokens) == self.max_seq_length:
                out.append(encoding)
            else:
                out.append(self._truncate(encoding))
        return out


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer on some raw text")
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser


def train(args, cfg) -> Tokenizer:
    log.info(f"Training tokenizer: {args}")

    out_dir = normalize_path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CalbertTokenizer(
        max_seq_length=cfg.training.max_seq_length, lowercase=cfg.vocab.lowercase
    )

    tokenizer.train(
        [str(normalize_path(args.input_file))],
        max_vocab_size=cfg.vocab.max_size,
        min_frequency=cfg.vocab.min_frequency,
    )

    tokenizer.save(str(out_dir), f"ca.bpe.{len(tokenizer)}")

    log.info(
        f"Saved tokenizer as {out_dir}/ca.bpe.{len(tokenizer)}[-vocab.json|-merges.txt]"
    )

    return tokenizer
