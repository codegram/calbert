import logging
import pytest
import tempfile
import glob
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from calbert import tokenizer

texts = [
    "Porto posat l'esquinç al peu sense sutura marejant metges i perdius i això no es cura.",
    "La sang s’ha cuit fins a tornar-se dura i passa el temps i passa i això no es cura.",
    "Camí de massa ampla tessitura estintolada, encara sobre la corda insegura.",
]


@pytest.fixture(scope="module")
def input_file_and_outdir() -> (str, str):
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as input_file:
        with tempfile.TemporaryDirectory() as outdir:
            for text in texts:
                input_file.write(text + "\n")
            filename = input_file.name
            input_file.flush()
            yield filename, outdir


@pytest.mark.describe("tokenizer.train")
class TestTrainTokenizer:
    def _train(
        self, input_file_and_outdir: (str, str)
    ) -> (tokenizer.CalbertTokenizer, str):
        input_file, outdir = input_file_and_outdir
        args, cfg = self._args_and_cfg(input_file, outdir)

        return tokenizer.train(args, cfg), outdir

    def _args_and_cfg(
        self, input_file: str, outdir: str
    ) -> (argparse.Namespace, OmegaConf):
        args = tokenizer.arguments().parse_args(
            ["--input-file", input_file, "--out-dir", outdir]
        )
        config = [
            "vocab.size=10",
            "vocab.min_frequency=2",
            "vocab.lowercase=True",
        ]
        cfg = OmegaConf.from_dotlist(config)
        return args, cfg

    @pytest.mark.it("Trains a tokenizer on some corpus")
    def test_train(self, input_file_and_outdir):
        t, outdir = self._train(input_file_and_outdir)

        assert len(t) == 38
        assert t.id_to_token(0) == "<unk>"
        assert t.id_to_token(1) == "<pad>"
        assert t.id_to_token(2) == "[MASK]"
        assert t.id_to_token(3) == "[SEP]"
        assert t.id_to_token(4) == "[CLS]"

    @pytest.mark.it("Encodes single sentences BERT-style with CLS and SEP")
    def test_single_sentence_encoding(self, input_file_and_outdir):
        t, outdir = self._train(input_file_and_outdir)
        encoded = t.process("hola com anem")
        assert t.id_to_token(encoded.ids[0]) == "[CLS]"
        assert t.id_to_token(encoded.ids[-1]) == "[SEP]"

    @pytest.mark.it("Encodes pairs of sentences BERT-style with CLS and SEP")
    def test_sentence_pair_encoding(self, input_file_and_outdir):
        t, outdir = self._train(input_file_and_outdir)
        encoded = t.process("hola com anem", "molt be i tu")
        assert t.id_to_token(encoded.ids[0]) == "[CLS]"
        assert t.id_to_token(encoded.ids[15]) == "[SEP]"
        assert t.id_to_token(encoded.ids[-1]) == "[SEP]"

    @pytest.mark.it("Truncates encodings to a max sequence length")
    def test_truncates_encodings(self, input_file_and_outdir):
        t, outdir = self._train(input_file_and_outdir)

        encoded = t.process("hola com anem", max_seq_len=12)
        assert encoded.tokens[-1] == "a"

    @pytest.mark.it("Pads encodings up to a max sequence length")
    def test_pads_encodings(self, input_file_and_outdir):
        t, outdir = self._train(input_file_and_outdir)

        encoded = t.process("hola com anem", max_seq_len=100)
        assert encoded.tokens[-1] == "<pad>"

    @pytest.mark.it("Saves the tokenizer's vocab and merges")
    def test_saves_tokenizer(self, input_file_and_outdir):
        _, outdir = self._train(input_file_and_outdir)

        assert glob.glob(outdir + "/*") == [
            outdir + "/ca.bpe.38-vocab.json",
            outdir + "/ca.bpe.38-merges.txt",
        ]
