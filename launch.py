#!/bin/python

import subprocess
import sys

lrank = sys.argv[1].split("=")[1]

batch_size = 512

s = f"python -m calbert train_model --tokenizer-dir /root/tokenizer --dataset-dir /root/dataset --out-dir /root/model --wandb-dir /root/wandb --train-batch-size {batch_size} --eval-batch-size {batch_size} --subset 1.0 --local-rank {lrank}"

subprocess.run(s.split(" "))
