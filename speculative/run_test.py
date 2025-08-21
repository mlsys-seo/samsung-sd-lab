import os, sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib
from utils import LLMEngine
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--cache-dir", type=str, default="/workspace/cache", help="Cache directory for models.")
    parser.add_argument("--output-dir", type=str, default="./results", help="Result directory for models.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get('WORLD_SIZE', 1)))
    parser.add_argument("--min-output-length", type=int, default=1)
    parser.add_argument("--max-output-length", type=int, default=157)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--problem", type=str, default=None)
    return parser.parse_args()

def check_args(args):
    assert args.tensor_parallel_size <= args.world_size, f"tensor_parallel_size {args.tensor_parallel_size} must be less than or equal to world_size {args.world_size}"
    assert args.pipeline_parallel_size <= args.world_size, f"pipeline_parallel_size {args.pipeline_parallel_size} must be less than or equal to world_size {args.world_size}"
    assert args.local_rank < args.tensor_parallel_size*args.pipeline_parallel_size, f"local_rank {args.local_rank} must be less than parallel_size {args.tensor_parallel_size*args.pipeline_parallel_size}"
    assert args.world_size == args.tensor_parallel_size*args.pipeline_parallel_size, f"world_size {args.world_size} must be equal to tensor_parallel_size {args.tensor_parallel_size} * pipeline_parallel_size {args.pipeline_parallel_size}"
    assert args.batch_size == 1, f"batch_size {args.batch_size} must be 1 for single-token inference"

if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    print(f"Problem {args.problem} Execute")
    if args.draft_model is not None:
        if args.problem:
            module_name = f"utils.spec_engine_{args.problem}"
            LLMSpecEngine = getattr(importlib.import_module(module_name), "LLMSpecEngine")
        else:
            from utils.spec_engine import LLMSpecEngine
        engine = LLMSpecEngine(args)
    else:
        engine = LLMEngine(args)

    engine.warmup()


    prompt_text = "Write a short story: Once upon a time,"
    prompt_text = engine.tokenizer.apply_chat_template([
        {
            "role": "user",
            "content": prompt_text
        }
    ], tokenize=False, add_generation_prompt=True)

    output_text, stats = engine.generate(prompt_text)
    print(f"overall latency : {stats['overall_latency']:.2f} secs")
    print(f"time to output token : {stats['time_to_output_token']:.2f} ms")
    print(f"prefill throughput : {stats['prefill_throughput']:.2f} tokens/second")
    print(f"generation throughput : {stats['generation_throughput']:.2f} tokens/second")
