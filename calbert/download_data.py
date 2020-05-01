import argparse
import logging
from pathlib import Path
import urllib.request
import os
import math

from calbert.utils import normalize_path

log = logging.getLogger(__name__)


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OSCAR dataset")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="The folder where to store the raw data",
    )
    parser.add_argument(
        "--force-download",
        type=bool,
        default=False,
        help="Whether to redownload the dataset even if it is already there",
    )
    parser.add_argument(
        "--force-split",
        type=bool,
        default=False,
        help="Whether to split the dataset even if it is already split",
    )
    return parser


def run(args, cfg):
    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force_download and (out_dir / "dataset.txt.gz").exists():
        log.info("Raw compressed dataset already exists --all good!")
    else:
        log.warning("Downloading raw compressed dataset")

        urllib.request.urlretrieve(
            "https://traces1.inria.fr/oscar/files/Compressed/ca_dedup.txt.gz",
            out_dir / "dataset.txt.gz",
        )

    if (
        not args.force_split
        and (out_dir / "train.txt").exists()
        and (out_dir / "valid.txt").exists()
    ):
        log.info("Dataset is already split into train/valid --all good!")
    else:
        log.info("Calculating dataset size")
        n = int(
            os.popen(f"gunzip -c {str(out_dir)}/dataset.txt.gz | wc -l").read().strip()
        )
        training_size = math.floor(n * (1 - cfg.data.valid_split) / 1.0)
        valid_size = n - training_size
        log.info(
            f"Splitting dataset in {training_size} training examples and {valid_size} validation examples ({cfg.data.valid_split * 100}%)"
        )
        os.system(
            f"gunzip -c {str(out_dir)}/dataset.txt.gz | split -l {training_size} - {str(out_dir)}/ && mv {str(out_dir)}/aa {str(out_dir)}/train.txt && mv {str(out_dir)}/ab {str(out_dir)}/valid.txt"
        )
