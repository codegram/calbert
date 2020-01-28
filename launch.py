#!/bin/python

import subprocess
import sys

lrank = sys.argv[1].split("=")[1]

batch_size = 22

s = f"python -m calbert train_model --tokenizer-dir /root/tokenizer --dataset-dir /root/dataset --out-dir /root/model --wandb-dir /root/wandb --train-batch-size {batch_size} --eval-batch-size {batch_size * 2} --subset 0.001 --local-rank {lrank}"

subprocess.run(s.split(" "))
