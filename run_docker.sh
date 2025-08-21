#!/usr/bin/env bash
set -euo pipefail

# Docker 컨테이너 실행 스크립트
# 사용법: ./run_docker.sh [container_name]

# 컨테이너 이름 (기본값: autodist-container)
CONTAINER_NAME="${1:-autodist-container}"

# 이미지 (필요 시 변경)
IMAGE="nvcr.io/nvidia/pytorch:23.10-py3"

# 호스트 볼륨 경로 계산(../ 를 절대경로로)
HOST_WORKDIR="$(cd .. && pwd)"

echo "🚀 AutoTP Docker 컨테이너를 시작합니다..."
echo "📦 컨테이너 이름: ${CONTAINER_NAME}"
echo "🐳 이미지: ${IMAGE}"
echo "💾 볼륨: ${HOST_WORKDIR} -> /workspace/"

# 동일 이름의 기존 컨테이너가 있으면 '존재'만 알리고 종료(제거하지 않음)
EXISTING_IDS="$(docker ps -aq -f "name=^${CONTAINER_NAME}$" || true)"
if [ -n "${EXISTING_IDS}" ]; then
  echo "⚠️  기존 컨테이너가 존재합니다: ${CONTAINER_NAME}"
  exit 1
fi

# 캐시 디렉토리 준비
mkdir -p "${HOST_WORKDIR}/cache"

# 새 컨테이너 실행 (종료 시 자동 삭제를 원치 않으면 --rm 제거)
docker run -it --rm \
  --gpus all \
  --ipc=host \
  --net=host \
  --name "${CONTAINER_NAME}" \
  -v ${PWD}:/workspace/sd-lab \
  "${IMAGE}" \
  bash

echo "✅ 컨테이너가 종료되었습니다."
