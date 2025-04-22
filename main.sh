#!/bin/bash

# CUDA 디바이스 설정
export CUDA_VISIBLE_DEVICES=0

# 공통 인자
IMG_SIZE=336
BATCH_SIZE=16
NUM_WORKERS=8
SEED=42
JSON_PATH_BASE="base_classes"
JSON_PATH_TASKS="5classes_tasks"

# # BASE 학습 (task_id=0)
# echo "[INFO] Start BASE PHASE"
# python main.py \
#     --task_id 0 \
#     --meta_epochs 50 \
#     --gpu 0 \
#     --img_size $IMG_SIZE \
#     --batch_size $BATCH_SIZE \
#     --num_workers $NUM_WORKERS \
#     --seed $SEED \
#     --json_path $JSON_PATH_BASE

# CONTINUAL 학습 (task_id=1~5)
for TASK_ID in {1..12}; do
    echo "[INFO] Start CONTINUAL PHASE - Task $TASK_ID"
    python main.py \
        --task_id $TASK_ID \
        --meta_epochs 20 \
        --gpu 0 \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --json_path $JSON_PATH_TASKS
done
