#!/bin/python

import subprocess
import sys

lrank = sys.argv[1].split('=')[1]

s = f"python -m calbert train_model --tokenizer-dir /root/tokenizer --dataset-dir /root/dataset --out-dir /root/model --tensorboard-dir /root/tensorboard --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --subset 1.0 --wandb True --local-rank {lrank}"

subprocess.run(s.split(' '))