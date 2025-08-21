# AutoDist: Automatic Tensor & Pipeline Parallelism

> **GitHub Repository**: [https://github.com/JaeminK/samsung_scheduling.git](https://github.com/JaeminK/samsung_scheduling.git)

## 프로젝트 개요

AutoDist는 단일 GPU HuggingFace 모델을 자동으로 Tensor Parallel과 Pipeline Parallel 모델로 변환하는 프로젝트입니다. 이 프로젝트는 분산 학습의 핵심 개념들을 구현하는 4가지 과제로 구성되어 있습니다.

## 🎯 구현 과제

이 프로젝트는 다음과 같은 4가지 핵심 과제로 구성되어 있습니다:



### Problem 1: Pipeline Parallelism 구현
- **PipelineParallelTransformerLayer**: 스테이지 간 데이터 전송을 위한 send/recv 연산
- 스테이지의 첫 번째/마지막 레이어에서만 통신 수행

### Problem 2: Pipeline Parallel Stage Construction 구현
- 각 레이어를 PipelineParallelTransformerLayer로 래핑
- 스테이지 내에서의 위치 정보 설정

### Problem 3: Tensor Parallelism 구현
- **ColumnParallelLinear**: 입력을 여러 GPU로 분할하여 병렬 처리
- **RowParallelLinear**: 각 GPU가 출력의 일부를 계산하고 all_reduce로 합산

### Problem 4: Tensor Parallel Stage Construction 구현
- **Attention Layer**: Query/Key/Value는 Column Parallel, Output은 Row Parallel
- **MLP Layer**: Up projection은 Column Parallel, Down projection은 Row Parallel



## 📁 프로젝트 구조

```
samsung_scheduling/
├── src/autoDist/
│   ├── layer.py          # Problem 1, 3 구현 위치
│   ├── utils.py          # Problem 2, 4 구현 위치
│   └── solutions.md      # 모든 문제의 정답과 상세 설명
├── benchmarks/           # 테스트 스크립트
│   ├── test_single.sh    # 단일 GPU 테스트
│   ├── test_tp_dist.sh   # Tensor Parallel 분산 테스트 (torchrun)
│   ├── test_pp_dist.sh   # Pipeline Parallel 분산 테스트 (torchrun)
│   ├── run_tp_rank0.sh   # Tensor Parallel Rank 0 디버깅용
│   ├── run_tp_rank1.sh   # Tensor Parallel Rank 1 디버깅용
│   ├── run_pp_rank0.sh   # Pipeline Parallel Rank 0 디버깅용
│   └── run_pp_rank1.sh   # Pipeline Parallel Rank 1 디버깅용
├── main.py              # 메인 실행 파일
├── requirements.txt     # 의존성 패키지
├── run_docker.sh        # Docker 컨테이너 실행 스크립트
├── setup_container.sh   # 컨테이너 내부 설정 스크립트
└── README.md           # 이 파일
```

## 🚀 설치 및 실행

### 0. 개인 디렉토리 생성
```bash
mkdir {디렉토리 이름}
cd {디렉토리 이름}
```

### 1. Docker를 사용한 설치

**Docker 컨테이너 실행:**
```bash
# 저장소 클론
git clone https://github.com/JaeminK/samsung_scheduling.git
cd samsung_scheduling

# 컨테이너 이름 지정하여 실행
# 저번 실습과 다른 도커 이름 지정
./run_docker.sh {nickname}
```

### 2. 컨테이너 내부 설정

Docker 컨테이너 내부에서 다음 명령어를 실행하세요:

```bash
cd /workspace/samsung_scheduling

# 컨테이너 설정 스크립트 실행
./setup_container.sh
```

### 3. 단일 GPU 테스트 (모델 다운로드)

```bash
cd benchmarks
bash test_single.sh
```

### 4. Tensor Parallel 분산 테스트

```bash
cd benchmarks
bash test_tp_dist.sh
```

### 5. Pipeline Parallel 분산 테스트

```bash
cd benchmarks
bash test_pp_dist.sh
```

### 6. 디버깅용 테스트 (개별 GPU 실행)

**Tensor Parallel 디버깅:**
```bash
cd benchmarks
# Rank 0 실행 (pdb 디버깅 가능)
./run_tp_rank0.sh

# Rank 1 실행 (다른 터미널에서)
./run_tp_rank1.sh
```

**Pipeline Parallel 디버깅:**
```bash
cd benchmarks
# Rank 0 실행 (pdb 디버깅 가능)
./run_pp_rank0.sh

# Rank 1 실행 (다른 터미널에서)
./run_pp_rank1.sh
```

> **참고**: `run_*_rank*.sh` 스크립트는 pdb 디버깅을 위한 개별 GPU 실행용이고, `test_*_dist.sh` 스크립트는 `torchrun`을 사용한 자동 분산 실행용입니다.

### 7. GPU 설정 변경

모든 스크립트에서 사용할 GPU를 변경하려면:

```bash
# 스크립트 상단의 CUDA_VISIBLE_DEVICES 값을 수정
export CUDA_VISIBLE_DEVICES=0,1  # GPU 0, 1 사용
export CUDA_VISIBLE_DEVICES=2,3  # GPU 2, 3 사용
export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU 0, 1, 2, 3 사용
```

**주의**: `CUDA_VISIBLE_DEVICES`에 지정한 GPU 개수와 `--nproc_per_node` 값이 일치해야 합니다.

## 📋 요구사항

- Python 3.8+
- PyTorch 2.7.0
- Transformers
- CUDA 지원 GPU (분산 학습용)

## 🎓 학습 목표

이 프로젝트를 통해 다음을 학습할 수 있습니다:

1. **Tensor Parallelism**: 모델의 가중치를 여러 GPU에 분산하여 메모리 효율성 향상
2. **Pipeline Parallelism**: 모델을 여러 스테이지로 분할하여 처리량 향상
3. **분산 통신**: GPU 간 효율적인 데이터 전송 및 동기화
4. **모델 병렬화**: Transformer 아키텍처의 효율적인 분산 처리


---

**참고**: 구현 과제의 정답은 `src/autoDist/solutions.md` 파일을 참조하세요.

