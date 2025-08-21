# AutoDist 구현 과제 정답

이 문서는 AutoDist의 5가지 구현 과제에 대한 정답과 상세한 설명을 포함합니다.

## 구현 요약

### Problem 1: Pipeline Parallelism
- 스테이지 간 데이터 전송을 위한 send/recv 연산 구현
- 스테이지의 첫 번째/마지막 레이어에서만 통신 수행

### Problem 2: Stage Construction
- 각 레이어를 PipelineParallelTransformerLayer로 래핑
- 스테이지 내에서의 위치 정보 설정

### Problem 3: Tensor Parallelism
- **ColumnParallelLinear**: 입력을 여러 GPU로 분할하여 병렬 처리, 필요시 all_gather로 결과 수집
- **RowParallelLinear**: 각 GPU가 출력의 일부를 계산하고 all_reduce로 합산

### Problem 4: Layer Parallelization
- **Attention**: Query/Key/Value는 Column Parallel, Output은 Row Parallel
- **MLP**: Up projection은 Column Parallel, Down projection은 Row Parallel

## Problem 1: Pipeline Parallel Transformer Layer Implementation

```python
def forward(self, *args, **kwargs):
    # Problem 1: Pipeline Parallel Transformer Layer Implementation
    """
    구현해야 할 내용:
    1. 현재 스테이지가 첫 번째 스테이지가 아니고, 이 레이어가 스테이지의 첫 번째 레이어인 경우:
       - 이전 스테이지로부터 입력을 받아야 합니다
       - self.comm_utils.recv(hidden_states, self.stage_num - 1) 사용
       - 받은 hidden_states로 args를 업데이트하세요
    
    2. 레이어의 forward 연산을 수행하세요
       - self.layer(*args, **kwargs) 호출
    
    3. 현재 스테이지가 마지막 스테이지가 아니고, 이 레이어가 스테이지의 마지막 레이어인 경우:
       - 다음 스테이지로 출력을 전송해야 합니다
       - self.comm_utils.send(output, self.stage_num + 1) 사용
    
    함수 사용법:
    - self.comm_utils.recv(tensor, src_rank): src_rank로부터 텐서를 수신
    - self.comm_utils.send(tensor, dst_rank): dst_rank로 텐서를 전송
    - self.layer(*args, **kwargs): 실제 transformer 레이어의 forward 연산
    
    조건 확인:
    - is_first_layer_in_stage: 스테이지의 첫 번째 레이어인지 확인
    - is_last_layer_in_stage: 스테이지의 마지막 레이어인지 확인
    - stage_num: 현재 스테이지 번호 (0부터 시작)
    
    반환값: 레이어의 출력 (마지막 레이어가 아닌 경우 다음 스테이지로 전송됨)
    """
    hidden_states = args[0]
    if self.is_first_layer_in_stage and self.stage_num > 0:
        hidden_states = self.comm_utils.recv(hidden_states, self.stage_num - 1)
        args = (hidden_states,) + args[1:]

    output = self.layer(*args, **kwargs)
    
    if self.is_last_layer_in_stage and self.stage_num < self.comm_utils.world_size - 1:
        self.comm_utils.send(output, self.stage_num + 1)
    return output
```

## Problem 2: Pipeline Parallel Stage Construction Implementation

```python
# Problem 2: Pipeline Parallel Stage Construction Implementation
"""
구현해야 할 내용:
Pipeline Parallel을 위한 스테이지별 레이어 래핑을 구현하세요.

1. 현재 레이어가 스테이지의 첫 번째 레이어인지 확인:
   - i == stage_info['layers_start'] 조건으로 확인
   - is_first_layer_in_stage 변수에 저장

2. 현재 레이어가 스테이지의 마지막 레이어인지 확인:
   - i == stage_info['layers_end'] - 1 조건으로 확인
   - is_last_layer_in_stage 변수에 저장

3. PipelineParallelTransformerLayer로 레이어를 래핑:
   - layer: 원본 transformer 레이어
   - comm_utils: 통신 유틸리티 (None으로 설정)
   - stage_num: 현재 스테이지 번호
   - is_first_layer_in_stage: 스테이지 첫 번째 레이어 여부
   - is_last_layer_in_stage: 스테이지 마지막 레이어 여부
   - layer_idx: 스테이지 내에서의 레이어 인덱스 (i - stage_info['layers_start'])

4. 래핑된 레이어를 stage_layers 리스트에 추가

함수 사용법:
- PipelineParallelTransformerLayer(layer, comm_utils, stage_num, 
  is_first_layer_in_stage, is_last_layer_in_stage, layer_idx)
- stage_info['layers_start']: 스테이지의 시작 레이어 인덱스
- stage_info['layers_end']: 스테이지의 끝 레이어 인덱스 (exclusive)

주의사항: 
- 스테이지의 첫 번째 레이어는 이전 스테이지로부터 입력을 받아야 함
- 스테이지의 마지막 레이어는 다음 스테이지로 출력을 전송해야 함
- layer_idx는 스테이지 내에서의 상대적 위치를 나타냄
"""
is_first_layer_in_stage = (i == stage_info['layers_start'])
is_last_layer_in_stage = (i == stage_info['layers_end'] - 1)
wrapped_layer = PipelineParallelTransformerLayer(
    layer, None, stage_num, is_first_layer_in_stage, is_last_layer_in_stage, i-stage_info['layers_start']
)
stage_layers.append(wrapped_layer)
```

## Problem 3: Column Parallel Linear Layer Implementation

### ColumnParallelLinear.forward()

```python
def forward(self, input: torch.Tensor):
    # Problem 3: Column Parallel Linear Layer Implementation
    """
    구현해야 할 내용:
    1. 입력 텐서 input에 대해 선형 변환을 수행하세요
       - torch.nn.functional.linear(input, self.weight) 사용
    2. bias가 존재하는 경우 bias를 더하세요
       - self.bias가 None이 아닌 경우에만 수행
    3. gather_output이 True인 경우에만 all_gather 연산을 수행하세요
       - self.comm_utils.all_gather(output) 사용하여 모든 GPU의 출력을 수집
    
    함수 사용법:
    - torch.nn.functional.linear(input, weight): 입력과 가중치의 선형 변환
    - self.comm_utils.all_gather(tensor): 모든 GPU의 텐서를 수집하여 배치 차원으로 연결
    
    반환값: 최종 출력 텐서 (gather_output=True인 경우 모든 GPU의 출력이 연결된 텐서)
    """
    output = torch.nn.functional.linear(input, self.weight)
    if self.bias is not None:
        output += self.bias
    if self.gather_output:
        output = self.comm_utils.all_gather(output)
    return output
```

### RowParallelLinear.forward()

```python
def forward(self, input: torch.Tensor):
    # Problem 3: Row Parallel Linear Layer Implementation
    """
    구현해야 할 내용:
    1. 입력 텐서 input에 대해 선형 변환을 수행하세요
       - torch.nn.functional.linear(input, self.weight) 사용
    2. all_reduce 연산을 수행하여 모든 GPU의 출력을 합산하세요
       - self.comm_utils.all_reduce(output) 사용
    3. bias가 존재하는 경우 bias를 더하세요
       - self.bias가 None이 아닌 경우에만 수행
    
    함수 사용법:
    - torch.nn.functional.linear(input, weight): 입력과 가중치의 선형 변환
    - self.comm_utils.all_reduce(tensor): 모든 GPU의 텐서를 합산 (element-wise sum)
    
    주의사항: Row Parallel에서는 각 GPU가 출력의 일부를 계산하므로, 
    all_reduce를 통해 모든 GPU의 결과를 합산해야 합니다.
    
    반환값: 모든 GPU의 출력이 합산된 최종 텐서
    """
    output = torch.nn.functional.linear(input, self.weight)
    output = self.comm_utils.all_reduce(output)
    if self.bias is not None:
        output += self.bias
    return output
```



## Problem 4: Attention Layer Parallelization Implementation

```python
if _is_self_attention(name):
    # Problem 4: Attention Layer Parallelization Implementation
    """
    구현해야 할 내용:
    Attention 레이어의 Query, Key, Value, Output projection을 병렬화하세요.
    
    1. Query, Key, Value projection (입력 projection):
       - _is_attention_projection(name) 함수로 확인
       - ColumnParallelLinear 사용 (입력을 여러 GPU로 분할)
       - weights[name]['weight'][rank]로 현재 GPU의 가중치 가져오기
       - bias가 있는 경우 weights[name]['bias'][rank] 사용
       - comm_utils는 None으로 설정 (이미 분할된 가중치 사용)
    
    2. Output projection (출력 projection):
       - Query, Key, Value가 아닌 경우 (else 블록)
       - RowParallelLinear 사용 (출력을 여러 GPU로 분할)
       - weights[name]['weight'][rank]로 현재 GPU의 가중치 가져오기
       - 원본 모듈의 bias 사용 (module.bias)
       - comm_utils는 None으로 설정
    
    함수 사용법:
    - _is_attention_projection(name): Query/Key/Value projection인지 확인
    - ColumnParallelLinear(weight, bias, comm_utils): 입력을 분할하는 병렬 선형 레이어
    - RowParallelLinear(weight, bias, comm_utils): 출력을 분할하는 병렬 선형 레이어
    - weights[name]['weight'][rank]: 현재 GPU(rank)의 분할된 가중치
    - weights[name]['bias'][rank]: 현재 GPU(rank)의 분할된 bias
    
    주의사항: Attention의 입력 projection은 Column Parallel, 출력 projection은 Row Parallel을 사용합니다.
    """
    if _is_attention_projection(name):
        split_weight_tensor = weights[name]['weight'][rank]
        split_bias_tensor = None
        if hasattr(module, "bias") and module.bias is not None:
            split_bias_tensor = weights[name]['bias'][rank]
        modules_to_replace[name] = ColumnParallelLinear(split_weight_tensor, split_bias_tensor, None)
    else: # This is output projection layer
        split_weight_tensor = weights[name]['weight'][rank]
        modules_to_replace[name] = RowParallelLinear(split_weight_tensor, module.bias, None)
```

## Problem 4: MLP Layer Parallelization Implementation

```python
else:
    # Problem 4: MLP Layer Parallelization Implementation
    """
    구현해야 할 내용:
    MLP 레이어의 Up projection과 Down projection을 병렬화하세요.
    
    1. Up projection (첫 번째 선형 레이어):
       - _is_mlp_up_projection(name) 함수로 확인
       - ColumnParallelLinear 사용 (입력을 여러 GPU로 분할)
       - weights[name]['weight'][rank]로 현재 GPU의 가중치 가져오기
       - bias가 있는 경우 weights[name]['bias'][rank] 사용
       - comm_utils는 None으로 설정
    
    2. Down projection (두 번째 선형 레이어):
       - Up projection이 아닌 경우 (else 블록)
       - RowParallelLinear 사용 (출력을 여러 GPU로 분할)
       - weights[name]['weight'][rank]로 현재 GPU의 가중치 가져오기
       - 원본 모듈의 bias 사용 (module.bias)
       - comm_utils는 None으로 설정
    
    함수 사용법:
    - _is_mlp_up_projection(name): MLP의 첫 번째 선형 레이어인지 확인
    - ColumnParallelLinear(weight, bias, comm_utils): 입력을 분할하는 병렬 선형 레이어
    - RowParallelLinear(weight, bias, comm_utils): 출력을 분할하는 병렬 선형 레이어
    - weights[name]['weight'][rank]: 현재 GPU(rank)의 분할된 가중치
    - weights[name]['bias'][rank]: 현재 GPU(rank)의 분할된 bias
    
    주의사항: MLP의 첫 번째 레이어(Up projection)는 Column Parallel, 
    두 번째 레이어(Down projection)는 Row Parallel을 사용합니다.
    """
    if _is_mlp_up_projection(name):
        split_weight_tensor = weights[name]['weight'][rank]
        split_bias_tensor = None
        if hasattr(module, "bias") and module.bias is not None:
            split_bias_tensor = weights[name]['bias'][rank]
        modules_to_replace[name] = ColumnParallelLinear(split_weight_tensor, split_bias_tensor, None)
    else: # This is down projection layer
        split_weight_tensor = weights[name]['weight'][rank]
        modules_to_replace[name] = RowParallelLinear(split_weight_tensor, module.bias, None)
```



