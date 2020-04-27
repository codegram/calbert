
import logging
import argparse
import collections
from pathlib import Path

import sentencepiece as spm
from transformers import AlbertTokenizer

from .utils import normalize_path

log = logging.getLogger(__name__)


def load(cfg, vocab_path: Path) -> AlbertTokenizer:
    return AlbertTokenizer(str(vocab_path.absolute()), keep_accents=True, do_lower_case=cfg.vocab.lowercase)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tokenizer on some raw text")
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-threads", type=int, default=32)
    return parser


def train(args, cfg) -> AlbertTokenizer:
    log.info(f"Training tokenizer: {args}")

    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = cfg.vocab.max_size

    name = [str(out_dir) + '/ca']
    name.append("uncased" if cfg.vocab.lowercase else "cased")
    name.append(str(vocab_size))
    prefix = ".".join(name)

    log.info(f'Will save to {prefix}')

    rule = '_cf' if cfg.vocab.lowercase else ''

    cmd = f"--num_threads={args.num_threads} --normalization_rule_name=nmt_nfkc{rule} --input={str(args.input_file.absolute())} --model_prefix={prefix} --vocab_size={vocab_size} --pad_id=0 --unk_id=1 --eos_id=-1 --bos_id=-1 --control_symbols=[CLS],[SEP],[MASK] --user_defined_symbols=(,),',\",-,.,–,£,€,$,·,´ --shuffle_input_sentence=true --input_sentence_size=5000000 --character_coverage=0.99995 --model_type=unigram"

    spm.SentencePieceTrainer.Train(cmd)

    return prefix
