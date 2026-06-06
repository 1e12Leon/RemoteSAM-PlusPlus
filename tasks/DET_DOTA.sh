#!/bin/bash

FILE="./result/DET_DOTA/DET_DOTA.log"
mkdir -p "$(dirname "$FILE")"
: > "$FILE"

MODAL=${MODAL:-rgb}
RESUME=${RESUME:-"checkpoints/DINOv3-convnext-B-modernbert-lora/model_last_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"./refer/data/DOTAv2_patches/test-dev/images"}
ANNO_FOLDER=${ANNO_FOLDER:-"./refer/data/DOTAv2_patches/test-dev/annfiles"}
REF_FILE_PATH=${REF_FILE_PATH:-""}
INSTANCES_PATH=${INSTANCES_PATH:-""}

CATEGORY=("airplane" "ship" "storagetank" "baseballfield" "tenniscourt" "basketballcourt" "groundtrackfield" "harbor" "bridge" "large-vehicle" "small-vehicle" "helicopter" "roundabout" "soccer-ball-field" "swimming-pool" "container-crane" "airport" "helipad")

for i in "${!CATEGORY[@]}"; do
    python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 \
        ./tasks/code/eval/DET_DOTA.py \
        --resume "$RESUME" \
        --split test \
        --save_path "./result/DET_DOTA/" \
        --task "DET" \
        --dataset "DOTAv2" \
        --modal "$MODAL" \
        --imageFolder "$IMAGE_FOLDER" \
        --annoFolder "$ANNO_FOLDER" \
        --ref_file_path "$REF_FILE_PATH" \
        --instances_path "$INSTANCES_PATH" \
        --_class "${CATEGORY[$i]}" \
        2>&1 | tee -a $FILE
done
