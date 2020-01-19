import spell.client
import argparse
from pathlib import Path
import logging

log = logging.getLogger(__name__)

reqs = open((Path(__file__) / "../../requirements.txt").resolve()).readlines()

packages = [dep.strip() for dep in reqs if "==" in dep]


def wait(client, run, logs=False):
    if run is None:
        return
    if logs:
        for line in run.logs():
            if line.status == client.runs.RUNNING and not line.status_event:
                log.info(f"[{run.id}] {line}")
    run.wait_status(client.runs.COMPLETE)
    run.refresh()


def download_data(client, cfg):
    filename = "ca_dedup.txt.gz" if cfg.dedup else "ca.txt.gz"
    data_url = f"https://traces1.inria.fr/oscar/files/Compressed/{filename}"
    r = client.runs.new(command=f"wget -O data.txt.gz {data_url}", idempotent=True)
    log.info(
        f"[{r.id}] Downloading data... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/data.txt.gz")


def splitting_dataset(
    client, cfg, data_cfg, raw_data_path, forced_run_id=None,
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


def create_tokenizer(client, cfg, data_path, forced_run_id=None):
    if forced_run_id:
        log.info(f"[{forced_run_id}] Training tokenizer... (forcedly cached)")
        return (None, f"runs/{forced_run_id}/tokenizer")

    r = client.runs.new(
        command="mkdir -p /spell/tokenizer && python -m calbert train_tokenizer --input-file /spell/train.txt --out-dir /spell/tokenizer",
        commit_label="repo",
        machine_type="cpu-big",
        pip_packages=packages,
        attached_resources={f"{data_path}/train.txt": "train.txt"},
        idempotent=True,
    )
    log.info(
        f"[{r.id}] Training tokenizer... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/tokenizer")


def create_dataset(client, cfg, data_path, tokenizer_path, forced_run_id=None):
    if forced_run_id:
        log.info(f"[{forced_run_id}] Creating dataset... (forcedly cached)")
        return (None, f"runs/{forced_run_id}/dataset")

    r = client.runs.new(
        command="mkdir -p $PWD/dataset && python -m calbert dataset --train-file $PWD/train.txt --valid-file $PWD/valid.txt --tokenizer-dir $PWD/tokenizer --out-dir $PWD/dataset",
        commit_label="repo",
        machine_type="cpu-big",
        pip_packages=packages,
        attached_resources={
            f"{data_path}/train.txt": "train.txt",
            f"{data_path}/valid.txt": "valid.txt",
            tokenizer_path: "tokenizer",
        },
        idempotent=True,
    )
    log.info(
        f"[{r.id}] Creating dataset... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/dataset")


def train_model(client, cfg, tokenizer_path, dataset_path):
    r = client.runs.new(
        command=" ".join(
            [
                # "git clone https://www.github.com/nvidia/apex &&",
                # """cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && """,
                # "cd .. && rm -fr apex &&",
                "pip install -r requirements.txt && ",
                "python",
                "-m calbert",
                "train_model",
                "--tokenizer-dir",
                "$PWD/tokenizer",
                "--dataset-dir",
                "$PWD/dataset",
                "--out-dir",
                "$PWD/model",
                "--tensorboard-dir",
                "$PWD/tensorboard",
                "--fp16",
                "--wandb",
            ]
        ),
        tensorboard_directory="tensorboard",
        commit_label="repo",
        docker_image="codegram/apex-pytorch1.4-cuda10.1:latest",
        machine_type=cfg.training.machine_type,
        # pip_packages=packages,
        attached_resources={tokenizer_path: "tokenizer", dataset_path: "dataset"},
        # idempotent=True,
    )
    log.info(
        f"[{r.id}] Training model... ({'cached' if r.already_existed else 'running'})"
    )
    return (r, f"runs/{r.id}/model")


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the whole workflow on spell.run")
    return parser


def run(args, cfg):
    client = spell.client.from_environment()

    raw_data_run, raw_data_path = download_data(client, cfg.data)
    wait(client, raw_data_run)

    data_run, data_path = splitting_dataset(
        client, cfg.training, cfg.data, raw_data_path
    )
    wait(client, data_run)

    tokenizer_run, tokenizer_path = create_tokenizer(
        client, cfg, data_path, forced_run_id=314
    )
    wait(client, tokenizer_run)

    dataset_run, dataset_path = create_dataset(
        client, cfg, data_path, tokenizer_path, forced_run_id=315
    )
    wait(client, dataset_run)

    model_run, model_path = train_model(client, cfg, tokenizer_path, dataset_path)
    wait(client, model_run, logs=True)
