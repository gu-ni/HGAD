#!/bin/bash

# CUDA 디바이스 설정
export CUDA_VISIBLE_DEVICES=0

# phase: base 또는 continual
PHASE=$1

# 공통 인자
IMG_SIZE=336
BATCH_SIZE=16
NUM_WORKERS=8
SEED=42
JSON_PATH="5classes_tasks"

# 실행 분기
if [ "$PHASE" = "base" ]; then
    echo "Start [BASE PHASE]"
    python main.py \
        --phase base \
        --meta_epochs 50 \
        --gpu 0 \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --json_path $JSON_PATH

elif [ "$PHASE" = "continual" ]; then
    echo "Start [CONTINUAL PHASE]"
    python main.py \
        --phase continual \
        --meta_epochs 20 \
        --gpu 0 \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        --seed $SEED \
        --json_path $JSON_PATH

else
    echo "에러: 유효하지 않은 phase입니다. 'base' 또는 'continual' 중 하나를 선택하세요."
fi
