import os, sys
import time
import argparse

import torch
import torch.distributed as dist

import transformers
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ModelRunner, DistributedSampler

            
class LLMEngine:
    def __init__(self, args):
        # set seed
        torch.manual_seed(args.seed)
        transformers.set_seed(args.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        self.args = args
        self.device = f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu'
        self.use_cuda_graph = args.use_cuda_graph

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)

        self.model_runner = ModelRunner(
            model_name_or_path=args.model,
            device=self.device,
            dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            local_rank=args.local_rank,
            world_size=args.world_size,
            capture_graph=self.use_cuda_graph
        )

        eos_token_ids = self.get_eos_token_ids()

        self.sampler_manager = DistributedSampler(
            vocab_size=self.model_runner.model_config.vocab_size,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            local_rank=args.local_rank,
            world_size=args.world_size,
            eos_token_ids=eos_token_ids,
            max_output_length=args.max_output_length
        )

        self.stats = {}


    def get_eos_token_ids(self):
        if isinstance(self.tokenizer.eos_token_id, list):
            eos_token_ids = self.tokenizer.eos_token_id
        else:
            eos_token_ids = [self.tokenizer.eos_token_id]
        
        if isinstance(self.tokenizer.pad_token_id, list):
            pad_token_ids = self.tokenizer.pad_token_id
        else:
            pad_token_ids = [self.tokenizer.pad_token_id]
        
        gen_config_eos_id = self.model_runner.model.model.generation_config.eos_token_id
        if isinstance(gen_config_eos_id, list):
            eos_token_ids.extend(gen_config_eos_id)
        else:
            eos_token_ids.append(gen_config_eos_id)

        eos_token_ids.extend(pad_token_ids)
        eos_token_ids = list(set(eos_token_ids))
        return eos_token_ids


    def register_stat(self, *args, **kwargs):
        pass


    def step(self, input_ids, num_ouput_logits=1):
        logits = self.model_runner.run_model(input_ids=input_ids, num_output_logits=num_ouput_logits)
        tokens = self.sampler_manager.sample_and_distribute(logits)
        return tokens


    def generate(self, prompt):
        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        input_ids = prompt_ids.clone()
        output_ids = torch.tensor([], dtype=torch.long, device=self.device)
        stop_flag = False

        torch.cuda.synchronize()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        record_time = []
        time_to_first_token = 0
        torch.cuda.synchronize()
        
        if dist.is_initialized():
            dist.barrier()
        
        print(f"============================================\n")
        if self.args.local_rank == 0:
            print(prompt, end='', flush=True)

        while not stop_flag:
            start_time.record()

            tokens = self.step(input_ids)

            if self.sampler_manager.should_sample:
                output_ids, stop_flag = self.sampler_manager.post_process(tokens, output_ids)
            
            stop_flag = self.sampler_manager.broadcast_stop_flag(stop_flag, self.device)
            input_ids = tokens[:, -1:]
            
            end_time.record()
            torch.cuda.synchronize()
            if time_to_first_token == 0:
                time_to_first_token = start_time.elapsed_time(end_time)
            else:
                record_time.append(start_time.elapsed_time(end_time))
            
            if self.args.local_rank == 0:
                # new_tokens = tokens[0, -1].cpu().numpy()
                new_tokens = tokens[0,:].cpu().numpy()
                print(self.tokenizer.decode(new_tokens), end='', flush=True)
        
        torch.cuda.synchronize()
        print(f"\n============================================\n")
        if dist.is_initialized():
            dist.barrier()
        
        if self.sampler_manager.should_output_results():
            final_output = torch.cat([prompt_ids, output_ids.view(1, -1)], dim=1)
            record_time = record_time[10:] # remove first 10 records for graph capture warmup
        
        num_prefill_tokens = prompt_ids.shape[1]
        num_output_tokens = output_ids.shape[0]
        generation_time = sum(record_time)
        overall_latency = time_to_first_token + sum(record_time)

        stats = {
            "overall_latency": overall_latency / 1e3,
            "time_to_output_token": generation_time / num_output_tokens,
            "prefill_throughput": num_prefill_tokens / (time_to_first_token / 1e3),
            "generation_throughput": num_output_tokens / (generation_time / 1e3)
        }
        return final_output, stats


    def warmup(self):
        for _ in range(10):
            input_ids = torch.randint(0, self.model_runner.model_config.vocab_size, (1, 50)).to(self.device)
            tokens = self.step(input_ids)
            input_ids = tokens[:, -1:]
        