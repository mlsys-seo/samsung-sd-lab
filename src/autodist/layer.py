import torch

from .comm_utils import CommUtils

class TensorParallelEmbedding(torch.nn.Embedding):
    def __init__(self, weight: torch.Tensor, padding_idx: int, comm_utils: CommUtils):
        num_embeddings, embedding_dim = weight.shape
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.weight.copy_(weight)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output = self.comm_utils.all_gather(output)
        return output


class ColumnParallelLinear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, comm_utils: CommUtils, gather_output: bool = False):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=True if bias is not None else False, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.gather_output = gather_output
        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        import pdb; pdb.set_trace()
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


class RowParallelLinear(torch.nn.Linear):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, comm_utils: CommUtils):
        out_features, in_features = weight.shape
        super().__init__(in_features, out_features, bias=True if bias is not None else False, dtype=weight.dtype)
        self.comm_utils = comm_utils
        self.weight.copy_(weight)
        if bias is not None:
            self.bias.copy_(bias)

    def forward(self, input: torch.Tensor):
        import pdb; pdb.set_trace()
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


class PipelineParallelEmbedding(torch.nn.Module):
    def __init__(self, embedding: torch.nn.Module, dtype: torch.dtype, stage_num: int, is_first_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_first_stage = is_first_stage
        self.dtype = dtype
        
        if is_first_stage:
            self.embedding = embedding
        else:
            self.embedding_dim = embedding.embedding_dim
            self.num_embeddings = embedding.num_embeddings
            self.padding_idx = getattr(embedding, 'padding_idx', None)
    
    def forward(self, input_ids):
        if self.is_first_stage:
            return self.embedding(input_ids)
        else:
            return torch.empty(input_ids.shape[0], input_ids.shape[1], self.embedding_dim, dtype=self.dtype, device=input_ids.device)


class PipelineParallelFinalLayerNorm(torch.nn.Module):
    def __init__(self, final_layer_norm: torch.nn.Module, stage_num: int, is_last_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_last_stage = is_last_stage
        
        if is_last_stage:
            self.final_layer_norm = final_layer_norm
    
    def forward(self, hidden_states):
        if self.is_last_stage:
            return self.final_layer_norm(hidden_states)
        else:
            return hidden_states


class PipelineParallelLMHead(torch.nn.Module):
    def __init__(self, lm_head: torch.nn.Module, stage_num: int, is_last_stage: bool = False):
        super().__init__()
        self.stage_num = stage_num
        self.is_last_stage = is_last_stage
        
        if is_last_stage:
            self.lm_head = lm_head
        else:
            self.in_features = lm_head.in_features
            self.out_features = lm_head.out_features
            self.bias = lm_head.bias is not None

    def forward(self, hidden_states):
        if self.is_last_stage:
            return self.lm_head(hidden_states)
        else:
            return None


class PipelineParallelTransformerLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, comm_utils: CommUtils, stage_num: int, 
                 is_first_layer_in_stage: bool = False, is_last_layer_in_stage: bool = False, layer_idx: int = 0):
        super().__init__()
        self.layer = layer
        self.comm_utils = comm_utils
        self.stage_num = stage_num
        self.is_first_layer_in_stage = is_first_layer_in_stage
        self.is_last_layer_in_stage = is_last_layer_in_stage
        self.layer_idx = layer_idx
        
        self._adjust_layer_idx_for_attention()
    
    def _adjust_layer_idx_for_attention(self):
        for name, module in self.layer.named_modules():
            if hasattr(module, 'layer_idx'):
                setattr(module, 'layer_idx', self.layer_idx)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            layer = super().__getattr__("layer")
        except AttributeError:
            raise
        return getattr(layer, name)

    def forward(self, *args, **kwargs):
        import pdb; pdb.set_trace()
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