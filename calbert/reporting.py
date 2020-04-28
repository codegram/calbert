import deepkit
from fastai2.basics import Recorder, Callback


class DeepkitCallback(Callback):
    "A `Callback` to report metrics to Deepkit"
    run_after = Recorder

    def __init__(self, args, cfg):
        super(DeepkitCallback).__init__()
        self.args = args
        self.cfg = cfg
        self.experiment: deepkit.Experiment = args.experiment
        self.total_examples = args.max_items or 19557475
        self.total_batches = int(self.total_examples / self.args.train_batch_size)
        self.batch = -1

    def begin_fit(self):
        # FIXME: look into why it doesn't work
        # self.experiment.watch_torch_model(self.learn.model)
        pass

    def begin_epoch(self):
        self.experiment.iteration(self.epoch, total=self.args.epochs)

    def begin_train(self):
        pass

    def begin_validate(self):
        pass

    def after_train(self):
        pass

    def after_validate(self):
        self._write_stats()

    def after_batch(self):
        self.batch += 1
        self.experiment.log_metric("train_loss", self.smooth_loss)
        self.experiment.log_metric("raw_loss", self.loss)
        self.experiment.batch(
            self.batch, size=self.args.train_batch_size, total=self.total_batches,
        )

    def after_epoch(self):
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
                print([n, s])
                self.experiment.log_metric(n, float(f"{s:.6f}"))
