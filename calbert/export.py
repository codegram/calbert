from pathlib import Path
from typing import Tuple, List
from shutil import copyfile
import argparse
import logging

import torch
from transformers import AlbertConfig, AlbertForMaskedLM

log = logging.getLogger(__name__)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained model as a standard HuggingFace Transformer"
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        required=True,
        help="The folder where ca.bpe.VOCABSIZE-vocab.json and ca.bpe.VOCABSIZE-merges.txt are",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=Path,
        help="Path to the trained model weights (usually bestmodel.pth).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        required=True,
        help="The output directory where the exported model will be written.",
    )
    return parser


def albert_config(cfg, args) -> AlbertConfig:
    return AlbertConfig(vocab_size=cfg.vocab.max_size, **dict(cfg.model))


def run(args, cfg):
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Exporting model config")
    model_config = albert_config(cfg, args)
    model_config.save_pretrained(args.out_dir)

    log.info("Exporting tokenizer")
    copyfile(next(args.tokenizer_dir.glob("*merges.txt")), args.out_dir / "merges.txt")
    copyfile(next(args.tokenizer_dir.glob("*vocab.json")), args.out_dir / "vocab.json")

    log.info("Exporting trained model weights")
    state = torch.load(args.model_path, torch.device("cpu"))
    m = AlbertForMaskedLM(model_config)
    m.load_state_dict(state, strict=True)

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = m.module if hasattr(m, "module") else m

    torch.save(model_to_save.state_dict(), args.out_dir / "pytorch_model.bin")
