import deepkit
import torch
import math

from fastai2.basics import Recorder, Callback, random, rank_distrib, num_distrib
from calbert.tokenizer import AlbertTokenizer
from calbert.model import CalbertForMaskedLM


class DeepkitCallback(Callback):
    "A `Callback` to report metrics to Deepkit"
    run_after = Recorder

    def __init__(self, args, cfg, tokenizer: AlbertTokenizer):
        super(DeepkitCallback).__init__()
        self.args = args
        self.cfg = cfg
        self.experiment: deepkit.Experiment = args.experiment
        self.tokenizer = tokenizer
        self.total_examples = args.max_items or 19557475
        self.log_every_batches = math.ceil(
            self.total_examples / 200 / self.args.train_batch_size
        )
        self.total_batches = math.ceil(self.total_examples / self.args.train_batch_size)
        self.n_preds = 4

    def begin_fit(self):
        # FIXME: look into why it doesn't work
        # self.experiment.watch_torch_model(self.learn.model)
        self.valid_dl = self.dls.valid.new(
            self.dls.valid_ds,
            bs=self.n_preds,
            rank=rank_distrib(),
            world_size=num_distrib(),
        )

    def begin_epoch(self):
        self.batch = 0
        self.experiment.iteration(self.epoch, total=self.args.epochs)

    def after_validate(self):
        self._write_stats()

    def begin_batch(self):
        pass

    def after_batch(self):
        if not self.learn.training:
            return

        self.batch += 1
        self.experiment.log_metric("train_loss", self.smooth_loss)
        self.experiment.log_metric("raw_loss", self.loss)
        self.experiment.batch(
            self.learn.train_iter,
            size=self.args.train_batch_size,
            total=self.total_batches,
        )
        if self.batch % self.log_every_batches == 0:  # log some insights
            b, _ = self.valid_dl.one_batch()
            with torch.no_grad():
                model = (
                    self.learn.model.module
                    if hasattr(self.learn.model, "module")
                    else self.learn.model
                )
                kls = model.__class__
                model.__class__ = CalbertForMaskedLM

                sources = [self.tokenizer.decode(x[0]).replace("<pad>", "") for x in b]
                masks = b[:, 1]
                filt = masks != -100
                labels = [
                    self.tokenizer.convert_ids_to_tokens(
                        masks[idx][filt[idx]], skip_special_tokens=False
                    )
                    for idx, f in enumerate(filt)
                ]

                _, prediction_scores = model(b)
                if prediction_scores.size(0) == 0:
                    return  # weird bug?
                predicteds = [
                    self.tokenizer.convert_ids_to_tokens(
                        torch.argmax(pscore[filt[i]], dim=1), skip_special_tokens=False
                    )
                    for i, pscore in enumerate(prediction_scores)
                ]
                insight = [
                    {
                        "text": source,
                        "correct+predicted": list(zip(labels[idx], predicteds[idx])),
                    }
                    for idx, source in enumerate(sources)
                ]
                self.experiment.log_insight(insight, name="predictions")
                model.__class__ = kls

    def after_epoch(self):
        self.experiment.iteration(self.epoch + 1, total=self.args.epochs)
        name = f"model_{self.epoch}"
        self.learn.save(name)
        self.experiment.add_output_file(str(self.learn.path / "models" / f"{name}.pth"))

    def after_fit(self):
        self.learn.save(f"final")
        self.experiment.add_output_file(str(self.learn.path / "models" / "final.pth"))

    def _write_stats(self):
        metric_names = list(self.recorder.metric_names).copy()
        values = list(self.recorder.log).copy()

        del metric_names[-1]

        if len(metric_names) - len(values) == 1:
            del metric_names[1]  # learn.validate() means there is no train_loss

        assert len(values) == len(metric_names)

        for n, s in zip(metric_names, values):
            if n not in ["epoch"]:
                self.experiment.log_metric(n, float(f"{s:.6f}"))
