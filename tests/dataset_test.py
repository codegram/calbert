import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from calbert import tokenizer

from .tokenizer_test import train_tokenizer


@pytest.mark.describe("dataset.Dataset")
class TestDataset:
    @pytest.mark.it("Turns raw text into a ready to consume dataset")
    def test_train(self, input_file_and_outdir):
        t, tokenizer_dir = train_tokenizer(input_file_and_outdir)

