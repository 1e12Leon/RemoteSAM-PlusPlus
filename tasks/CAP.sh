#!/bin/bash

FILE="./result/CAP/CAP.log"
mkdir -p "$(dirname "$FILE")"
: > "$FILE"

MODAL=${MODAL:-rgb}
RESUME=${RESUME:-"checkpoints/DINOv3-convnext-B-modernbert-lora/model_last_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"./refer/data/UCM_captions/Images"}
ANNO_FOLDER=${ANNO_FOLDER:-"./refer/data/UCM_captions/dataset.json"}
REF_FILE_PATH=${REF_FILE_PATH:-""}
INSTANCES_PATH=${INSTANCES_PATH:-""}

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
    ./tasks/code/eval/CAP.py \
    --resume "$RESUME" \
    --save_path "./result/CAP/" \
    --task "CAP" \
    --dataset "UCM" \
    --modal "$MODAL" \
    --imageFolder "$IMAGE_FOLDER" \
    --annoFolder "$ANNO_FOLDER" \
    --ref_file_path "$REF_FILE_PATH" \
    --instances_path "$INSTANCES_PATH" \
    2>&1 | tee -a $FILE
