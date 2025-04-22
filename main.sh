#!/bin/bash

# CUDA 디바이스 설정
export CUDA_VISIBLE_DEVICES=0

# 공통 인자
IMG_SIZE=336
BATCH_SIZE=16
NUM_WORKERS=8
SEED=42
JSON_PATH_BASE="base_classes_except_continual_ad"
TASK_JSON_GROUPS=(
    "5classes_tasks_except_continual_ad" \
    "10classes_tasks_except_continual_ad" \
    "30classes_tasks_except_continual_ad"
)
NUM_TASKS_GROUPS=(
    6 \
    3 \
    1
)

# BASE 학습 (task_id=0)
echo "[INFO] Start BASE PHASE"
python main.py \
    --task_id 0 \
    --meta_epochs 50 \
    --gpu 0 \
    --img_size $IMG_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --json_path $JSON_PATH_BASE

# CONTINUAL 학습 (task_id=1~5)
for ((i=0; i<${#TASK_JSON_GROUPS[@]}; i++)); do
    NUM_TASKS=${NUM_TASKS_GROUPS[$i]}
    for ((TASK_ID=1; TASK_ID<=NUM_TASKS; TASK_ID++)); do
        echo "[INFO] Start CONTINUAL PHASE - Task $TASK_ID"
        python main.py \
            --task_id $TASK_ID \
            --meta_epochs 20 \
            --gpu 0 \
            --img_size $IMG_SIZE \
            --batch_size $BATCH_SIZE \
            --num_workers $NUM_WORKERS \
            --seed $SEED \
            --json_path "${TASK_JSON_GROUPS[i]}"
    done
done
