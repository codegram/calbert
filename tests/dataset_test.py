import pytest

import random

from calbert.dataset import CalbertDataset, Tokenize, Mask, ignore, SentencePair
from fastai2.data.all import DataLoader, TfmdDL, Datasets, Transform, stop
from fastai2.text.data import TensorText
from fastai2.basics import L

# from fastai2.text.all import *
from .conftest import InputData, folder
from .tokenizer_test import train_tokenizer


@pytest.fixture(scope="module")
def dataset():
    with InputData("train") as train_file:
        yield train_file


@pytest.fixture(scope="module")
def tokenizer(which="train") -> (str, str):
    with InputData(which) as train_file:
        with folder() as outdir:
            yield train_tokenizer((train_file, outdir))[0]


@pytest.mark.describe("dataset.CalbertDataset")
class TestCalbertDataset:
    @pytest.mark.it("Returns pairs of sentences")
    def test_iter(self, dataset):
        ds = iter(CalbertDataset(dataset))
        assert next(ds) == SentencePair(
            "Porto posat l'esquinç al peu sense sutura marejant metges i perdius i això no es cura.",
            "D'altra banda tampoc he anat al metge.",
        )
        assert next(ds) == SentencePair(
            "Camí de massa ampla tessitura estintolada, encara sobre la corda insegura.",
            "Sens dubte.",
        )

    @pytest.mark.it("Returns pairs of sentences up to a limit")
    def test_iter_with_max_items(self, dataset):
        ds = iter(CalbertDataset(dataset, max_items=1))
        assert next(ds) == SentencePair(
            "Porto posat l'esquinç al peu sense sutura marejant metges i perdius i això no es cura.",
            "D'altra banda tampoc he anat al metge.",
        )
        try:
            next(ds)
            assert False
        except StopIteration:
            assert True


@pytest.mark.describe("dataset.Tokenization")
class TestTokenization:
    @pytest.mark.it("Returns tokenized pairs of sentences")
    def test_tokenize(self, dataset, tokenizer):
        ds = CalbertDataset(dataset)
        tfms = [Tokenize(tokenizer, max_seq_len=12)]
        train_ds = Datasets(ds, tfms=tfms)

        encoded = next(iter(train_ds))[0][0]
        assert train_ds.decode([TensorText(encoded)]) == ("port d'a",)


@pytest.mark.describe("dataset.Mask")
class TestMask:
    @pytest.mark.it("Masks tokens with a probability")
    def test_mask(self, dataset, tokenizer):
        ds = CalbertDataset(dataset)
        tfms = [Tokenize(tokenizer, max_seq_len=12), Mask(tokenizer, probability=1.0)]
        train_ds = Datasets(ds, tfms=[tfms, [ignore]])

        inputs, other = next(iter(train_ds))

        assert inputs[0].size(0) == 12
        assert tokenizer.mask_token_id in inputs[0]
