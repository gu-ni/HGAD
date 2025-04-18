#!/bin/bash

# CUDA 디바이스 설정
export CUDA_VISIBLE_DEVICES=0

# phase: base 또는 continual
PHASE=$1

# 공통 인자
IMG_SIZE=336
BATCH_SIZE=16
SEED=42
OUTPUT_DIR="./outputs"
BASE_JSON="base_classes"
TASK_JSON="5classes_tasks"
PRETRAINED_PATH="${OUTPUT_DIR}/HGAD_base_img.pt"

# 실행 분기
if [ "$PHASE" = "base" ]; then
    echo "[BASE PHASE] 시작"
    python main.py \
        --phase base \
        --meta_epochs 50 \
        --gpu 0 \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --base_json $BASE_JSON

elif [ "$PHASE" = "continual" ]; then
    echo "[CONTINUAL PHASE] 시작"
    python main.py \
        --phase continual \
         --meta_epochs 20 \
        --gpu 0 \
        --img_size $IMG_SIZE \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_dir $OUTPUT_DIR \
        --task_json $TASK_JSON \
        --pretrained_path $PRETRAINED_PATH

else
    echo "에러: 유효하지 않은 phase입니다. 'base' 또는 'continual' 중 하나를 선택하세요."
fi
