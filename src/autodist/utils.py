import torch
import torch.distributed as dist
import os
import json
import pickle
import time

from .comm_utils import CommUtils
from .layer import TensorParallelEmbedding, RowParallelLinear, ColumnParallelLinear
from .layer import PipelineParallelTransformerLayer, PipelineParallelEmbedding, PipelineParallelLMHead, PipelineParallelFinalLayerNorm
from .wrapper import TensorParallelWrapper, PipelineParallelWrapper


def _initialize_distributed(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, device_id=torch.device(f'cuda:{rank}'))
    torch.cuda.set_device(rank)

def _initialize_distributed_shard(source_rank, tensor_parallel_size):
    shard_group = dist.new_group(
        ranks=list(range(source_rank, source_rank + tensor_parallel_size))
    )
    return CommUtils(shard_group)

def _get_last_part(name):
    return name.split(".")[-1]

def _get_second_last_part(name):
    parts = name.split(".")
    return parts[-2] if len(parts) > 1 else None

def _contains_pattern(name, patterns):
    return any(pattern in _get_last_part(name) for pattern in patterns)

def _is_position_embedding(name):
    return _get_last_part(name) == "embed_positions"

def _is_self_attention(name):
    return _get_second_last_part(name) == "self_attn"

def _is_lm_head(name):
    return _get_last_part(name) == "lm_head"

def _is_attention_projection(name):
    return _contains_pattern(name, ["q_proj", "k_proj", "v_proj"])

def _is_mlp_up_projection(name):
    return _contains_pattern(name, ["up_proj", "gate_proj"])

def _is_transformer_layer(name):
    return _contains_pattern(name, ['layers', 'blocks', 'transformer'])

def _is_layer_norm(name):
    return "norm" in _get_last_part(name)

def _is_embedding(module):
    return isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag))

def _is_linear(module):
    return isinstance(module, torch.nn.Linear)


def convert_tp_model_config(model: torch.nn.Module, world_size: int = 1):
    adjust_config_name = ["embed_dim", "hidden_size", "num_heads", "num_key_value_heads"]
    for _, module in model.named_modules():
        for config_name in adjust_config_name:
            if hasattr(module, config_name):
                config_value = getattr(module, config_name)
                setattr(module, config_name, config_value // world_size)
    
    adjust_config_name = ["n_embed", "hidden_size", "num_attention_heads", "num_key_value_heads"]
    for config_name in adjust_config_name:
        if hasattr(model.config, config_name):
            config_value = getattr(model.config, config_name)
            setattr(model.config, config_name, config_value // world_size)
    return model, model.config


def get_split_weights(model: torch.nn.Module, world_size: int):
    split_weights = {}
    for name, module in model.named_modules():
        if _is_embedding(module):
            assert module.max_norm is None or module.norm_type < 2
            if not _is_position_embedding(name):
                split_weights[name] = {}
                split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
        elif _is_linear(module):
            split_weights[name] = {}
            if _is_self_attention(name):
                if _is_attention_projection(name):
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
                    if hasattr(module, "bias") and module.bias is not None:
                        split_weights[name]['bias'] = torch.tensor_split(module.bias, world_size, dim=0)
                else: # This is output projection layer
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
            elif _is_lm_head(name):
                split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
            else:
                if _is_mlp_up_projection(name):
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=0)
                    if hasattr(module, "bias") and module.bias is not None:
                        split_weights[name]['bias'] = torch.tensor_split(module.bias, world_size, dim=0)
                else: # This is down projection layer
                    split_weights[name]['weight'] = torch.tensor_split(module.weight, world_size, dim=1)
    return split_weights


def convert_to_tensor_parallel(model: torch.nn.Module, weights: dict, rank: int = 0):
    modules_to_replace = {}
    for name, module in model.named_modules():
        if _is_embedding(module):
            if not _is_position_embedding(name):
                split_weight_tensor = weights[name]['weight'][rank]
                modules_to_replace[name] = TensorParallelEmbedding(split_weight_tensor, module.padding_idx, None)
        elif _is_linear(module):
            if _is_self_attention(name):
                import pdb; pdb.set_trace()
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
            elif _is_lm_head(name):
                split_weight_tensor = weights[name]['weight'][rank]
                modules_to_replace[name] = ColumnParallelLinear(split_weight_tensor, module.bias, None, gather_output=True)
            else:
                import pdb; pdb.set_trace()
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
    
    for name, new_module in modules_to_replace.items():
        module_path = name.split('.')
        parent_module = model
        for part in module_path[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, module_path[-1], new_module)
    
    return model


def get_tp_cache_path(model_name_or_path: str, cache_dir: str, tensor_parallel_size: int, rank: int):
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    tp_cache_dir = os.path.join(cache_dir, "tp_shards")
    return os.path.join(tp_cache_dir, f"{model_name}_tp_{tensor_parallel_size}", f"rank_{rank}")


def get_tp_completion_file(model_name_or_path: str, cache_dir: str, tensor_parallel_size: int):
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    tp_cache_dir = os.path.join(cache_dir, "tp_shards")
    return os.path.join(tp_cache_dir, f"{model_name}_tp_{tensor_parallel_size}", "tp_creation_complete")


def save_tp_shard_to_cache(model, cache_path: str, rank: int):
    os.makedirs(cache_path, exist_ok=True)
    torch.save(model, os.path.join(cache_path, "model.pt"))
    
    print(f"[Rank {rank}]: Saved TP shard to {cache_path}")


def load_tp_shard_from_cache(cache_path: str, device, execute_rank: int, tensor_parallel_size: int, capture_graph: bool = True):
    model = torch.load(os.path.join(cache_path, "model.pt"), weights_only=False, map_location='cpu')
    model = model.to(device)
    
    comm_utils = _initialize_distributed_shard(execute_rank, tensor_parallel_size)
    wrapped_model = TensorParallelWrapper(model, comm_utils, capture_graph=capture_graph)
    
    return wrapped_model, model.config


def create_and_save_tp_shards(model, model_name_or_path: str, cache_dir: str, 
                             tensor_parallel_size: int, execute_rank: int, 
                             dtype, device, local_rank: int, capture_graph: bool = True):
    completion_file = get_tp_completion_file(model_name_or_path, cache_dir, tensor_parallel_size)
    
    print(f"Creating TP shards by rank {local_rank}")
    
    try:
        split_weights = get_split_weights(model, tensor_parallel_size)
        model, _ = convert_tp_model_config(model, tensor_parallel_size)
        for rank in range(execute_rank, execute_rank + tensor_parallel_size):
            rank_model = convert_to_tensor_parallel(model, split_weights, rank - execute_rank)
            
            cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, rank)
            save_tp_shard_to_cache(rank_model, cache_path, rank)
        
        with open(completion_file, 'w') as f:
            f.write('complete')
        print(f"TP shards created successfully")
        
        current_cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, local_rank)
        return load_tp_shard_from_cache(current_cache_path, device, execute_rank, tensor_parallel_size, capture_graph)
        
    except Exception as e:
        print(f"Error creating TP shards: {e}")
        raise


@torch.no_grad()
def load_or_create_tp_model(model_name_or_path: str, cache_dir: str, tensor_parallel_size: int, 
                           execute_rank: int, dtype, device, local_rank: int, world_size: int, capture_graph: bool = True):
    if not dist.is_initialized():
        _initialize_distributed(local_rank, world_size)
        print(f"Initialized distributed training on rank {local_rank} with world size {world_size}")
    
    tp_cache_path = get_tp_cache_path(model_name_or_path, cache_dir, tensor_parallel_size, local_rank)
    completion_file = get_tp_completion_file(model_name_or_path, cache_dir, tensor_parallel_size)
    
    if os.path.exists(tp_cache_path) and os.path.exists(completion_file):
        print(f"[Rank {local_rank}]: Loading TP shard from cache: {tp_cache_path}")
        return load_tp_shard_from_cache(tp_cache_path, device, execute_rank, tensor_parallel_size, capture_graph)
        
    if local_rank == execute_rank:
        print(f"[Rank {local_rank}]: Creating TP shards for tensor_parallel_size={tensor_parallel_size}")
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype
            )
        
        adjust_config_names = ["n_embed", "hidden_size", "num_attention_heads", "num_key_value_heads"]
        for config_name in adjust_config_names:
            if hasattr(model.config, config_name):
                config_value = getattr(model.config, config_name)
                if config_value % tensor_parallel_size != 0 or config_value < tensor_parallel_size:
                    print(f"[Rank {local_rank}]: {config_name} {config_value} is not divisible by tensor_parallel_size {tensor_parallel_size}")
                    exit()
        
        model, config = create_and_save_tp_shards(model, model_name_or_path, cache_dir, 
                                                 tensor_parallel_size, execute_rank, 
                                                 dtype, device, local_rank, capture_graph)
        
        print(f"[Rank {local_rank}]: Loaded TP shard from cache: {tp_cache_path}")
        if model is not None:
            return model, config
    
    print(f"[Rank {local_rank}]: Waiting for execute_rank to create TP shards...")
    while not os.path.exists(completion_file):
        time.sleep(0.1)
    
    model, config = load_tp_shard_from_cache(tp_cache_path, device, execute_rank, tensor_parallel_size, capture_graph)
    print(f"[Rank {local_rank}]: Loaded TP shard from cache: {tp_cache_path}")
    
    return model, config


def get_split_stage(model: torch.nn.Module, stage_num: int, pipeline_parallel_size: int):
    stage_info = {
        'stage_num': stage_num,
        'is_first_stage': stage_num == 0,
        'is_last_stage': stage_num == pipeline_parallel_size - 1,
        'layers_start': 0,
        'layers_end': 0,
    }
    
    transformer_layers = None
    for name, module in model.named_modules():
        if any(key in name.split(".")[-1] for key in ['layers', 'blocks', 'transformer']):
            if isinstance(module, torch.nn.ModuleList):
                transformer_layers = module
                break
    
    if transformer_layers is None:
        raise ValueError("Could not find transformer layers in the model")
    
    num_layers = len(transformer_layers)
    layers_per_stage = num_layers // pipeline_parallel_size
    
    stage_info['layers_start'] = stage_num * layers_per_stage
    stage_info['layers_end'] = (stage_num + 1) * layers_per_stage if stage_num < pipeline_parallel_size - 1 else num_layers
    
    return stage_info


def convert_to_pipeline_parallel(model: torch.nn.Module, stage_info: dict, stage_num: int):
    from copy import deepcopy
    
    stage_model = deepcopy(model)
    modules_to_replace = {}
    
    embedding_layers = []
    embedding_layers_name = []
    for name, module in stage_model.named_modules():
        if _is_embedding(module):
            embedding_layers.append(module)
            embedding_layers_name.append(name)
    
    transformer_layers = None
    transformer_layers_name = None
    for name, module in stage_model.named_modules():
        if _is_transformer_layer(name):
            transformer_layers = module
            transformer_layers_name = name
            break
    
    last_layer_norm = None
    last_layer_norm_name = None
    for name, module in stage_model.named_modules():
        if _is_layer_norm(name):
            last_layer_norm = module
            last_layer_norm_name = name
    
    last_linear = None
    last_linear_name = None
    for name, module in stage_model.named_modules():
        if _is_linear(module):
            last_linear = module
            last_linear_name = name
    
    if len(embedding_layers) > 0:
        for embedding_layer, embedding_layer_name in zip(embedding_layers, embedding_layers_name):
            wrapped_embedding = PipelineParallelEmbedding(
                embedding_layer, model.config.torch_dtype, stage_num, stage_info['is_first_stage']
            )
            modules_to_replace[embedding_layer_name] = wrapped_embedding
    else:
        raise ValueError("Could not find embedding layer in the model")
    
    if transformer_layers is not None:
        stage_layers = torch.nn.ModuleList()
        for i in range(stage_info['layers_start'], stage_info['layers_end']):
            layer = transformer_layers[i]
            import pdb; pdb.set_trace()
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
        modules_to_replace[transformer_layers_name] = stage_layers
    else:
        raise ValueError("Could not find transformer layers in the model")
    
    if last_layer_norm is not None:
        wrapped_final_layer_norm = PipelineParallelFinalLayerNorm(
            last_layer_norm, stage_num, stage_info['is_last_stage']
        )
        modules_to_replace[last_layer_norm_name] = wrapped_final_layer_norm
    else:
        raise ValueError("Could not find last layer norm in the model")
    
    if last_linear is not None:
        wrapped_lm_head = PipelineParallelLMHead(
            last_linear, stage_num, stage_info['is_last_stage']
        )
        modules_to_replace[last_linear_name] = wrapped_lm_head
    else:
        raise ValueError("Could not find last linear layer in the model")
    
    for name, new_module in modules_to_replace.items():
        module_path = name.split('.')
        parent_module = stage_model
        for part in module_path[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, module_path[-1], new_module)
    
    return stage_model


def convert_pp_model_config(model: torch.nn.Module, pipeline_parallel_size: int):
    """Convert model config for pipeline parallelism."""
    if hasattr(model.config, "num_hidden_layers"):
        layers_per_stage = model.config.num_hidden_layers // pipeline_parallel_size
        model.config.num_hidden_layers = layers_per_stage
    
    layer_count_attrs = ["n_layer", "num_layers", "n_layers"]
    for attr in layer_count_attrs:
        if hasattr(model.config, attr):
            layers_per_stage = getattr(model.config, attr) // pipeline_parallel_size
            setattr(model.config, attr, layers_per_stage)
    
    return model, model.config


def get_pp_cache_path(model_name_or_path: str, cache_dir: str, pipeline_parallel_size: int, stage_num: int, rank_in_stage: int):
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    pp_cache_dir = os.path.join(cache_dir, "pp_shards")
    return os.path.join(pp_cache_dir, f"{model_name}_pp_{pipeline_parallel_size}", f"stage_{stage_num}", f"rank_{rank_in_stage}")


def get_pp_completion_file(model_name_or_path: str, cache_dir: str, pipeline_parallel_size: int):
    model_name = os.path.basename(model_name_or_path.rstrip('/'))
    pp_cache_dir = os.path.join(cache_dir, "pp_shards")
    return os.path.join(pp_cache_dir, f"{model_name}_pp_{pipeline_parallel_size}", "pp_creation_complete")


def save_pp_shard_to_cache(model, cache_path: str, stage_num: int, rank_in_stage: int):
    os.makedirs(cache_path, exist_ok=True)
    torch.save(model, os.path.join(cache_path, "model.pt"))
    
    print(f"[Stage {stage_num}, Rank {rank_in_stage}]: Saved PP shard to {cache_path}")


def load_pp_shard_from_cache(cache_path: str, device, execute_rank: int, pipeline_parallel_size: int, capture_graph: bool = True, stage_num: int = 0):
    model = torch.load(os.path.join(cache_path, "model.pt"), weights_only=False, map_location='cpu')
    model = model.to(device)
    
    comm_utils = _initialize_distributed_shard(execute_rank, pipeline_parallel_size)
    wrapped_model = PipelineParallelWrapper(model, comm_utils, 
                                           capture_graph=capture_graph, stage_num=stage_num, rank_in_stage=0)
    
    return wrapped_model, model.config


def create_and_save_pp_shards(model, model_name_or_path: str, cache_dir: str, 
                             pipeline_parallel_size: int, execute_rank: int, 
                             dtype, device, local_rank: int, capture_graph: bool = True):
    completion_file = get_pp_completion_file(model_name_or_path, cache_dir, pipeline_parallel_size)
    
    print(f"Creating PP shards by rank {local_rank}")
    
    try:
        transformer_layers = None
        for name, module in model.named_modules():
            if any(key in name.split(".")[-1] for key in ['layers', 'blocks', 'transformer']):
                if isinstance(module, torch.nn.ModuleList):
                    transformer_layers = module
                    break
        
        if transformer_layers is None:
            raise ValueError("Could not find transformer layers in the model")
        
        num_layers = len(transformer_layers)
        if num_layers < pipeline_parallel_size:
            raise ValueError(f"Number of layers ({num_layers}) must be >= pipeline_parallel_size ({pipeline_parallel_size})")
        
        for stage_num in range(pipeline_parallel_size):
            stage_info = get_split_stage(model, stage_num, pipeline_parallel_size)
            stage_model = convert_to_pipeline_parallel(model, stage_info, stage_num)
            
            stage_model, _ = convert_pp_model_config(stage_model, pipeline_parallel_size)
            
            cache_path = get_pp_cache_path(model_name_or_path, cache_dir, pipeline_parallel_size, stage_num, 0)
            save_pp_shard_to_cache(stage_model, cache_path, stage_num, 0)
        
        with open(completion_file, 'w') as f:
            f.write('complete')
        print(f"PP shards created successfully")
        
        stage_num = local_rank
        current_cache_path = get_pp_cache_path(model_name_or_path, cache_dir, pipeline_parallel_size, stage_num, 0)
        return load_pp_shard_from_cache(current_cache_path, device, execute_rank, pipeline_parallel_size, capture_graph, stage_num)
        
    except Exception as e:
        print(f"Error creating PP shards: {e}")
        raise


@torch.no_grad()
def load_or_create_pp_model(model_name_or_path: str, cache_dir: str, pipeline_parallel_size: int, 
                           execute_rank: int, dtype, device, local_rank: int, world_size: int, capture_graph: bool = True):
    if not dist.is_initialized():
        _initialize_distributed(local_rank, world_size)
        print(f"Initialized distributed training on rank {local_rank} with world size {world_size}")
    
    stage_num = local_rank
    pp_cache_path = get_pp_cache_path(model_name_or_path, cache_dir, pipeline_parallel_size, stage_num, 0)
    completion_file = get_pp_completion_file(model_name_or_path, cache_dir, pipeline_parallel_size)
    
    if os.path.exists(pp_cache_path) and os.path.exists(completion_file):
        print(f"[Rank {local_rank}]: Loading PP shard from cache: {pp_cache_path}")
        return load_pp_shard_from_cache(pp_cache_path, device, execute_rank, pipeline_parallel_size, capture_graph, stage_num)
        
    if local_rank == execute_rank:
        print(f"[Rank {local_rank}]: Creating PP shards for pipeline_parallel_size={pipeline_parallel_size}")
        from transformers import AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except OSError:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=dtype
            )
        
        model, config = create_and_save_pp_shards(model, model_name_or_path, cache_dir, 
                                                 pipeline_parallel_size, execute_rank, 
                                                 dtype, device, local_rank, capture_graph)
        
        print(f"[Rank {local_rank}]: Loaded PP shard from cache: {pp_cache_path}")
        if model is not None:
            return model, config
    
    print(f"[Rank {local_rank}]: Waiting for execute_rank to create PP shards...")
    while not os.path.exists(completion_file):
        time.sleep(0.1)
    
    model, config = load_pp_shard_from_cache(pp_cache_path, device, execute_rank, pipeline_parallel_size, capture_graph, stage_num)
    print(f"[Rank {local_rank}]: Loaded PP shard from cache: {pp_cache_path}")
    
    return model, config
