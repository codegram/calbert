#!/bin/bash

spell run -v --machine-type v100 --mount runs/315/dataset:/spell/dataset --mount runs/314/tokenizer:/spell/tokenizer --from codegram/calbert:latest 'python -m calbert train_model --tokenizer-dir /spell/tokenizer --dataset-dir /spell/dataset --out-dir /spell/model --tensorboard-dir /spell/tensorboard --per_gpu_train_batch_size 128 --per_gpu_eval_batch_size 128 --subset 0.1 --wandb True'
