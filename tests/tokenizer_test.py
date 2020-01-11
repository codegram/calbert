import logging
import pytest
import tempfile
import glob
from pathlib import Path

from omegaconf import OmegaConf
from calbert import tokenizer

texts = [
    "Porto posat l'esquinç al peu sense sutura marejant metges i perdius i això no es cura.",
    "La sang s’ha cuit fins a tornar-se dura i passa el temps i passa i això no es cura.",
    "Camí de massa ampla tessitura estintolada, encara sobre la corda insegura.",
]


@pytest.mark.describe("tokenizer.train")
class TestTrainTokenizer:
    @pytest.mark.it("Trains a tokenizer on some corpus")
    def test_train(self):
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as fh:
            with tempfile.TemporaryDirectory() as outdir:
                for text in texts:
                    fh.write(text + "\n")
                input_file = fh.name
                fh.flush()

                args = tokenizer.arguments().parse_args(
                    ["--input-file", fh.name, "--out-dir", outdir]
                )
                config = [
                    "vocab.size=10",
                    "vocab.min_frequency=2",
                    "vocab.lowercase=True",
                ]
                cfg = OmegaConf.from_dotlist(config)
                tok, vocab_size = tokenizer.train(args, cfg)

                assert vocab_size == 56
                assert tok.id_to_token(0) == "[UNK]"
                assert tok.id_to_token(1) == "[SEP]"
                assert tok.id_to_token(2) == "[CLS]"

                example = (
                    "No tinc una massa ampla tessitura, encara que s'ha cuit en caça"
                )
                encoded = tok.encode(example)

                assert encoded.tokens[0] == "n"
                assert encoded.tokens[1] == "##o"

                assert glob.glob(outdir + "/*")[0] == outdir + "/ca.bert.56-vocab.txt"
