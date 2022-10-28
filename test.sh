#!/usr/bin/env bash

NGPUS=$1

torchrun --nproc_per_node=${NGPUS} test.py --launcher pytorch ${@:2}

# CUDA_VISIBLE_DEVICES=0,1 bash test.sh 2 --config *.yaml --set test_iter * batch_size *