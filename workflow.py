import spell.client
import hydra
import logging

log = logging.getLogger(__name__)

client = spell.client.from_environment()


def albert_requirements(gpu=False):
    return [
        f"tensorflow{'-gpu' if gpu else ''}==1.15",
        "tensorflow_hub==0.7",
        "sentencepiece",
    ]


def wait(run, logs=False):
    if run is None:
        return
    if logs:
        for line in run.logs():
            if line.status == client.runs.RUNNING and not line.status_event:
                log.info(f"[{run.id}] {line}")
    run.wait_status(client.runs.COMPLETE)
    run.refresh()


def download_data(cfg):
    filename = "ca_dedup.txt.gz" if cfg.dedup else "ca.txt.gz"
    data_url = f"https://traces1.inria.fr/oscar/files/Compressed/{filename}"
    r = client.runs.new(command=f"wget -O data.txt.gz {data_url}", idempotent=True)
    log.info(
        f"[{r.id}] Downloading data... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/data.txt.gz")


def download_sentencepiece_model(cfg):
    vocab_url = f"https://nlp.h-its.org/bpemb/ca/ca.wiki.bpe.vs{cfg.size}.vocab"
    model_url = f"https://nlp.h-its.org/bpemb/ca/ca.wiki.bpe.vs{cfg.size}.model"

    r = client.runs.new(
        command=f"wget -O ca.wiki.bpe.vocab {vocab_url} && wget -O ca.wiki.bpe.model {model_url}",
        idempotent=True,
    )
    log.info(
        f"[{r.id}] Downloading sentencepiece model... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/ca.wiki.bpe.vocab", f"runs/{r.id}/ca.wiki.bpe.model")


def create_pretraining_data(
    cfg, data_cfg, raw_data_path, forced_run_id=None,
):
    if forced_run_id:
        log.info(f"[{forced_run_id}] Splitting dataset... (forcedly cached)")
        return (None, f"runs/{forced_run_id}/dataset")

    cmd = f"""
    mkdir -p dataset && \
    len=$(gunzip -c data.txt.gz {f' | head -n {data_cfg.subset_size}' if data_cfg.subset else ''} | wc -l) && \
    training_size=$(echo 'import sys, math; length=int(sys.argv[1]); split=float(sys.argv[2]); print(math.floor(length * (1-split) / 1.))' | python - $len {data_cfg.valid_split}) && \
    gunzip -c data.txt.gz {f' | head -n {data_cfg.subset_size}' if data_cfg.subset else ''} | split -l $training_size - dataset/ && \
    mv dataset/aa dataset/train.txt && mv dataset/ab dataset/valid.txt
    """
    r = client.runs.new(
        command=cmd,
        attached_resources={raw_data_path: "data.txt.gz",},
        idempotent=True,
    )
    log.info(
        f"[{r.id}] Splitting dataset... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/dataset")


def pretrain(
    cfg,
    vocab_size,
    data_path,
    sentencepiece_vocab_path,
    sentencepiece_model_path,
    forced_run_id=None,
):
    if forced_run_id:
        log.info(f"[{forced_run_id}] Pretraining model... (forcedly cached)")
        return (None, f"runs/{forced_run_id}/model")

    cmd = f"""
    sed 's/"VOCAB_SIZE"/{vocab_size}/' config/{cfg.config}.json > config.json &&
    mkdir -p model &&
    python -m albert.run_pretraining \
        --input_file=output/*.tfrecord \
        --output_dir=model \
        --albert_config_file=config.json \
        --do_train \
        --do_eval \
        --train_batch_size={cfg.train_batch_size} \
        --eval_batch_size={cfg.eval_batch_size} \
        --max_seq_length={cfg.max_seq_length} \
        --max_predictions_per_seq={cfg.max_predictions_per_seq} \
        --optimizer='lamb' \
        --learning_rate=.00176 \
        --num_train_steps={cfg.num_train_steps} \
        --num_warmup_steps={cfg.num_warmup_steps} \
        --save_checkpoints_steps={cfg.save_checkpoints_steps}
    """
    r = client.runs.new(
        machine_type="v100",
        command=cmd,
        attached_resources={data_path: "output"},
        pip_packages=albert_requirements(gpu=True),
        envvars={"PYTHONPATH": "albert"},
        commit_label="repo",
        idempotent=True,  # does not work, using forced_run_id as a hack
    )
    log.info(
        f"[{r.id}] Pretraining model... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/model")


@hydra.main(config_path="config.yaml", strict=True)
def main(cfg):
    raw_data_run, raw_data_path = download_data(cfg.data)
    (
        sentencepiece_model_run,
        sentencepiece_vocab_path,
        sentencepiece_model_path,
    ) = download_sentencepiece_model(cfg.vocab)
    wait(raw_data_run)
    wait(sentencepiece_model_run)

    data_run, data_path = create_pretraining_data(
        cfg.pretraining, cfg.data, raw_data_path,
    )
    wait(data_run)

    if None:
        pretraining_run, pretraining_path = pretrain(
            cfg.pretraining,
            cfg.vocab.size,
            raw_data_path,
            sentencepiece_vocab_path,
            sentencepiece_model_path,
        )

        wait(pretraining_run, logs=true)

        log.info(f"Pretrained model is at {pretraining_path}")


if __name__ == "__main__":
    main()
