#! /bin/bash
# GPU 설정 (원하는 GPU 번호로 변경하세요)
clear
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DRAFT_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
MAX_OUTPUT_LENGTH=256

# nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./single --force-overwrite true \
python run_test.py \
    --model $MODEL_NAME \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length $MAX_OUTPUT_LENGTH \
    --use-cuda-graph \
    --problem C
# # nsys profile --trace=cuda,nvtx --cuda-graph=node -o ./single --force-overwrite true \
python run_test.py \
    --model $MODEL_NAME \
    --draft-model $DRAFT_MODEL_NAME \
    --num-draft-tokens 4 \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length $MAX_OUTPUT_LENGTH \
    --use-cuda-graph \
    --problem C