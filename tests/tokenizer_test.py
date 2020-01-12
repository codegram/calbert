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

                assert vocab_size == 38
                assert tok.id_to_token(0) == "<unk>"
                assert tok.id_to_token(1) == "<pad>"
                assert tok.id_to_token(2) == "[MASK]"
                assert tok.id_to_token(3) == "[SEP]"
                assert tok.id_to_token(4) == "[CLS]"

                encoded = tok.encode("hola com anem")

                assert tok.id_to_token(encoded.ids[0]) == "[CLS]"
                assert tok.id_to_token(encoded.ids[-1]) == "[SEP]"

                assert glob.glob(outdir + "/*") == [
                    outdir + "/ca.bpe.38-vocab.json",
                    outdir + "/ca.bpe.38-merges.txt",
                ]
