#!/bin/bash

FILE="./result/DET/DET.log"
mkdir -p "$(dirname "$FILE")"
: > "$FILE"

MODAL=${MODAL:-rgb}
RESUME=${RESUME:-"checkpoints/model_best_mIoU_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"/data/user2/zjj/DIORcoco/JPEGImages"}
ANNO_FOLDER=${ANNO_FOLDER:-"/data/user2/zjj/DIORcoco/Annotations/DIOR_test.json"}
REF_FILE_PATH=${REF_FILE_PATH:-""}
INSTANCES_PATH=${INSTANCES_PATH:-""}

CATEGORY=("airplane" "airport" "baseballfield" "basketballcourt" "bridge" "chimney" "dam" "expressway-service-area" "expressway-toll-station" "golffield" "groundtrackfield" "harbor" "overpass" "ship" "stadium" "storagetank" "tenniscourt" "trainstation" "vehicle" "windmill")

for i in "${!CATEGORY[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12349 \
        ./tasks/code/eval/DET.py \
        --resume "$RESUME" \
        --split test \
        --save_path "./result/DET/" \
        --task "DET" \
        --dataset "DIOR" \
        --modal "$MODAL" \
        --imageFolder "$IMAGE_FOLDER" \
        --annoFolder "$ANNO_FOLDER" \
        --ref_file_path "$REF_FILE_PATH" \
        --instances_path "$INSTANCES_PATH" \
        --_class "${CATEGORY[$i]}" \
        # --EPOC \
        2>&1 | tee -a $FILE
done
