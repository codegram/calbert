import hydra

from pathlib import Path
from functools import partial

from transformers import AlbertConfig, AlbertForSequenceClassification

import torch

from fastai2.basics import (
    accuracy,
    DataBlock,
    CategoryBlock,
    Learner,
    load_model,
    DataLoaders,
    TfmdDL,
    TfmdLists,
    L,
    CrossEntropyLossFlat,
    Transform,
    ToTensor,
    TransformBlock,
    get_text_files,
    RandomSplitter,
    to_device,
    params,
)
from fastai2.data.all import *
from fastai2.text.all import TensorText, TitledStr
from fastai2.callback.all import *

from calbert.training import TrainableAlbert
from calbert.tokenizer import CalbertTokenizer
from calbert.utils import normalize_path


class SentimentAnalysis(AlbertForSequenceClassification):
    def forward(self, inputs):
        x = inputs.permute(1, 0, 2)
        o = super().forward(input_ids=x[0], attention_mask=x[1], token_type_ids=x[2])
        logits = o[0]
        return logits


class Tokenize(Transform):
    def __init__(self, tokenizer: CalbertTokenizer):
        self.tokenizer = tokenizer

    def encodes(self, o: str):
        encoding = self.tokenizer.encode(o)
        return TensorText(
            torch.stack(
                [
                    torch.tensor(encoding.ids).long(),
                    torch.tensor(encoding.attention_mask).long(),
                    torch.tensor(encoding.type_ids).long(),
                ]
            )
        )

    def decodes(self, o: TensorText) -> str:
        decoded = (
            self.tokenizer.decode(o[0].tolist())
            .replace("[CLS]", "")
            .replace("[SEP]", "")
            .replace("<pad>", "")
            .strip()
        )
        return TitledStr(decoded)


def TextBlock(tokenizer: CalbertTokenizer):
    "`TextBlock` for single-label categorical targets"
    return TransformBlock(
        type_tfms=Tokenize(tokenizer=tokenizer)  # , batch_tfms=[ToTensor, Splat]
    )


def get_lines_from_files(path: Path):
    lines = []
    for filename in path.glob("*.txt"):
        kls = str(filename).split("/")[-1].replace(".txt", "")

        for idx, line in enumerate(open(filename, "r")):
            text = line.replace(". ", "").replace(" .", "").strip()
            lines.append((text, kls))
    return lines


def splitter(m: AlbertForSequenceClassification):
    return (
        L(m.albert.embeddings, m.albert.encoder,).map(params)
        + L(m.albert.pooler, m.albert.pooler_activation).map(params)
        + L(m.dropout, m.classifier).map(params)
    )


@hydra.main(config_path="./config/config.yaml", strict=True)
def main(cfg):
    tokenizer = CalbertTokenizer.from_dir(
        normalize_path(Path("tokenizer-cased")),
        max_seq_length=cfg.training.max_seq_length,
    )

    sentiment_data = DataBlock(
        blocks=(TextBlock(tokenizer=tokenizer), CategoryBlock),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_x=lambda x: x[0],
        get_y=lambda x: x[1],
        get_items=get_lines_from_files,
    )

    dsrc = sentiment_data.datasets(normalize_path(Path("clas_dataset")), verbose=True,)

    dls = dsrc.dataloaders()

    config = AlbertConfig(
        vocab_size=cfg.vocab.max_size, num_labels=dls.c, **dict(cfg.model)
    )

    model = SentimentAnalysis(config=config)

    model_to_resize = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))
    to_device(model)

    learn = Learner(
        dls=dls,
        model=model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
        splitter=splitter,
    )
    learn.path = normalize_path(Path("."))
    learn.model_dir = "model"
    learn.load("bestmodel", strict=False)

    learn.freeze()

    learn.fit_one_cycle(1, 1e-02)


if __name__ == "__main__":
    main()
