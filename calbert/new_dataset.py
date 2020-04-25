from pathlib import Path
import itertools
import re

import sentencepiece as spm
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm, trange
from transformers import AlbertTokenizer


def chunk(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # chunk('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    yield from itertools.zip_longest(*[iter(iterable)] * n, fillvalue=fillvalue)


punctuation = re.compile(r"[\.!\?]+")


def sentence_pairs(filename, min_length=8):
    with open(filename, encoding="utf-8") as f:
        for line in f:
            sentences = [
                s.strip()
                for s in punctuation.split(line)
                if len(s) >= min_length and " " in s
            ]
            for a, b in itertools.zip_longest(sentences[:-1], sentences[1:]):
                yield (a + ".", b + ".")


class CalbertDataset(TensorDataset):
    def __init__(
        self, tokenizer: AlbertTokenizer, dataset_path: Path, subset: float = 1.0,
    ):
        super(CalbertDataset, self).__init__()
        self.tokenizer = tokenizer
