#!/usr/bin/env bash

NGPUS=$1

torchrun --nproc_per_node=${NGPUS} train.py --launcher pytorch ${@:2}

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash train.sh 4 --config *.yaml