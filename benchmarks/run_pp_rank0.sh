#!/bin/bash

# GPU 설정 (원하는 GPU 번호로 변경하세요)
export CUDA_VISIBLE_DEVICES=0,1

# Set environment variables for rank 0
export LOCAL_RANK=0
export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run rank 0
# nsys profile --trace=cuda,nvtx -o pp_dist_rank0 \
python ../main.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir /workspace/cache \
    --output-dir ./results \
    --pipeline-parallel-size 2 \
    --local-rank 0 \
    --world-size 2 \
    --seed 1234 \
    --min-output-length 1 \
    --max-output-length 512
