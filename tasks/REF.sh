#!/bin/bash
FILE="./result/REF/REF-RMSIN.log"
mkdir -p "$(dirname "$FILE")"
: > "$FILE"

MODAL=${MODAL:-rgb}
RESUME=${RESUME:-"checkpoints/model_best_mIoU_RemoteSAM_DINOv3ConvNeXtB_ModernBERT.pth"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"/data/user2/zjj/RRSIS-D/images/rrsisd/JPEGImages"}
REF_FILE_PATH=${REF_FILE_PATH:-"/data/user2/zjj/RRSIS-D/rrsisd/refs(unc).p"}
INSTANCES_PATH=${INSTANCES_PATH:-"/data/user2/zjj/RRSIS-D/rrsisd/instances.json"}

# PYTHONPATH=. python ./tasks/code/eval/REF.py --nproc_per_node 1 --master_port 12345 \
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python ./tasks/code/eval/REF.py \
    --resume "$RESUME" \
    --modernbert_path ./gte-modernbert-base \
    --split test \
    --workers 4 \
    --img_size 896 \
    --modal "$MODAL" \
    --imageFolder "$IMAGE_FOLDER" \
    --ref_file_path "$REF_FILE_PATH" \
    --instances_path "$INSTANCES_PATH" \
    2>&1 | tee -a $FILE
