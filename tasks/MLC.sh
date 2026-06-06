#!/bin/bash

FILE="./result/MLC/MLC.log"
mkdir -p "$(dirname "$FILE")"
: > "$FILE"

MODAL=${MODAL:-rgb}
RESUME=${RESUME:-"checkpoints/model_best_mIoU_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"/data/user2/zjj/DIORcoco/JPEGImages"}
ANNO_FOLDER=${ANNO_FOLDER:-"/data/user2/zjj/DIORcoco/Annotations/DIOR_test.json"}
REF_FILE_PATH=${REF_FILE_PATH:-""}
INSTANCES_PATH=${INSTANCES_PATH:-""}
DIST_TIMEOUT=${DIST_TIMEOUT:-7200}

CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 \
    ./tasks/code/eval/MLC.py \
    --resume "$RESUME" \
    --split test \
    --save_path "./result/MLC/" \
    --task "MLC" \
    --dataset "DIOR" \
    --modal "$MODAL" \
    --imageFolder "$IMAGE_FOLDER" \
    --annoFolder "$ANNO_FOLDER" \
    --ref_file_path "$REF_FILE_PATH" \
    --instances_path "$INSTANCES_PATH" \
    --dist_timeout "$DIST_TIMEOUT" \
    2>&1 | tee -a $FILE
