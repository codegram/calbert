#!/bin/bash
set -e

git clone https://www.github.com/nvidia/apex
cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && cd ..
rm -fr apex