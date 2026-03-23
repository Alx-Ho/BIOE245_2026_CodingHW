#!/bin/bash

DATASET_ROOT="/data/pathmnist"
OUTPUT_ROOT="/scratch/hw"

python ./train_and_eval.py \
    --download \
    --output_root ${OUTPUT_ROOT} \
    --gpu_ids 0 \
    --dataset_root ${DATASET_ROOT}