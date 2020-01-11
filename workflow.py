import spell.client
import hydra
import logging

log = logging.getLogger(__name__)

packages = [
    dep.strip() for dep in open("./requirements.txt").readlines() if "==" in dep
]

client = spell.client.from_environment()


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
        command=cmd, attached_resources={raw_data_path: "data.txt.gz"}, idempotent=True,
    )
    log.info(
        f"[{r.id}] Splitting dataset... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/dataset")


def create_tokenizer(cfg, data_path):
    r = client.runs.new(
        command="mkdir -p /spell/tokenizer && python calbert.py tokenizer --input-file /spell/train.txt --out-dir /spell/tokenizer",
        commit_label="repo",
        pip_packages=packages,
        attached_resources={f"{data_path}/train.txt": "train.txt"},
        idempotent=True,
    )
    log.info(
        f"[{r.id}] Training tokenizer... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/tokenizer")


@hydra.main(config_path="config.yaml", strict=True)
def main(cfg):
    raw_data_run, raw_data_path = download_data(cfg.data)
    wait(raw_data_run)

    data_run, data_path = create_pretraining_data(
        cfg.pretraining, cfg.data, raw_data_path,
    )
    wait(data_run)

    vocab_run, vocab_path = create_tokenizer(cfg, data_path)

    wait(vocab_run, logs=True)


if __name__ == "__main__":
    main()
