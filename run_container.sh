#!/usr/bin/env bash
set -euo pipefail

########################################
# 1) Select profile
########################################
PROFILE=""   # "A" | "B" | "C" | "D" | "E"




########################################
# 2) Common settings
########################################
IMAGE="sd-lab:v0"
WORKDIR="/workspace"
MEM_LIMIT="32g"
SHM_SIZE="16g"
USE_IPC_HOST=1           # 1 = use --ipc=host
USE_ULIMIT_MEMLOCK=1     # 1 = use --ulimit memlock=-1:-1

# GPU assignment per user
GPU_A=0
GPU_B=1
GPU_C=2
GPU_D=3
GPU_E=4

########################################
# 3) Profile to GPU mapping
########################################
case "${PROFILE}" in
  "A") NAME="sd-gpu-A"; GPU_HOST_ID="${GPU_A}" ;;
  "B") NAME="sd-gpu-B"; GPU_HOST_ID="${GPU_B}" ;;
  "C") NAME="sd-gpu-C"; GPU_HOST_ID="${GPU_C}" ;;
  "D") NAME="sd-gpu-D"; GPU_HOST_ID="${GPU_D}" ;;
  "E") NAME="sd-gpu-E"; GPU_HOST_ID="${GPU_E}" ;;
  *) echo "[Error] PROFILE must be one of A/B/C/D/E."; exit 1 ;;
esac

########################################
# 4) Execution flags
########################################
IPC_FLAG=""; [[ "${USE_IPC_HOST}" -eq 1 ]] && IPC_FLAG="--ipc=host"
ULIMIT_FLAG=""; [[ "${USE_ULIMIT_MEMLOCK}" -eq 1 ]] && ULIMIT_FLAG="--ulimit memlock=-1:-1"

########################################
# 5) docker run
########################################
CMD=(docker run -it --rm
  --name "${NAME}"
  --gpus "device=${GPU_HOST_ID}"
  -e CUDA_VISIBLE_DEVICES=0
  -e TOKENIZERS_PARALLELISM=false
  --memory="${MEM_LIMIT}"
  --shm-size="${SHM_SIZE}"
  ${IPC_FLAG}
  ${ULIMIT_FLAG}
  -w "${WORKDIR}"
  "${IMAGE}"
  bash
)

echo "== PROFILE : ${PROFILE}"
echo "== GPU     : ${GPU_HOST_ID}"
echo "== Command to run =="
printf '%q ' "${CMD[@]}"; echo
exec "${CMD[@]}"