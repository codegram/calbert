import tempfile
import pytest

training_text = [
    "Porto posat l'esquinç al peu sense sutura marejant metges i perdius i això no es cura.",
    "La sang s’ha cuit fins a tornar-se dura i passa el temps i passa i això no es cura.",
    "Camí de massa ampla tessitura estintolada, encara sobre la corda insegura.",
]

validation_text = ["La corda insegura s'ha cuit"]


@pytest.fixture(scope="module")
def input_file_and_outdir(which="train") -> (str, str):
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as input_file:
        with tempfile.TemporaryDirectory() as outdir:
            for text in training_text if which == "train" else validation_text:
                input_file.write(text + "\n")
            filename = input_file.name
            input_file.flush()
            yield filename, outdir
