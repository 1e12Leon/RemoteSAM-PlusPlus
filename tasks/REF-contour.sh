#!/bin/bash
FILE="./result/REF/REF-contour.log"
: > $FILE


# PYTHONPATH=. python ./tasks/code/eval/REF.py --nproc_per_node 1 --master_port 12345 \
PYTHONPATH=. python ./tasks/code/eval/REF-contour.py \
    --swin_type base \
    --dataset rrsisd \
    --resume ./pretrained_weights/checkpoint.pth \
    --split val \
    --workers 4 \
    --window12 \
    --img_size 896 \
    --refer_data_root /data/user2/zjj/RRSIS-D \
    2>&1 | tee -a $FILE

