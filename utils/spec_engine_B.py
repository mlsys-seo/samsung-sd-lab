import torch

from utils.model_runner_C import ModelRunner
from utils.engine import LLMEngine

DEBUG = False

class LLMSpecEngine(LLMEngine):
    def __init__(self, args):
        super().__init__(args)

        assert args.num_draft_tokens > 0 and args.num_draft_tokens is not None, "num_draft_tokens must be greater than 0"
        assert args.draft_model is not None and args.draft_model != "", "draft_model must be specified"

        self.draft_model_runner = ModelRunner(
            model_name_or_path=args.draft_model,
            device=self.device,
            dtype=torch.bfloat16,
            cache_dir=args.cache_dir,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
        )
        self.num_draft_tokens = args.num_draft_tokens
        
        self.draft_tokens_buffer = torch.zeros((1, self.num_draft_tokens), dtype=torch.long, device=self.device)
        
        # TODO: Problem B :: implement this
        self.all_accepted = None
        self.last_bonus_token_id_buffer = None

    def get_accept_token_idx(self, draft_tokens: torch.Tensor, target_tokens: torch.Tensor):
        accept_token_idx = draft_tokens.shape[-1]
        for idx in range(self.num_draft_tokens):
            if draft_tokens[0, idx] != target_tokens[0, idx]:
                accept_token_idx = idx
                break
        return accept_token_idx

    
    def step(self, input_ids, num_ouput_logits=1):
        if DEBUG:
            print("-------------------------------- step --------------------------------")

        # prepare draft input ids
        # TODO: Problem B :: fix this
        draft_input_ids = input_ids.clone()

        # run draft model
        for draft_idx in range(self.num_draft_tokens):
            draft_logits = self.draft_model_runner.run_model(input_ids=draft_input_ids, num_output_logits=1)
            draft_token = self.sampler_manager.sample_and_distribute(draft_logits)
            draft_input_ids = draft_token
            self.draft_tokens_buffer[0, draft_idx].copy_(draft_token[0,-1])

        
        # run target model
        target_input_ids = torch.cat([input_ids, self.draft_tokens_buffer], dim=1)
        logits = self.model_runner.run_model(input_ids=target_input_ids, num_output_logits=self.num_draft_tokens+1)
        target_tokens = self.sampler_manager.sample_and_distribute(logits)


        # verify
        num_accepted_tokens = self.get_accept_token_idx(self.draft_tokens_buffer.cpu(), target_tokens.cpu())
        num_emitted_tokens = num_accepted_tokens + 1
        emitted_tokens = target_tokens[:, :num_emitted_tokens]

        # process bonus token
        # TODO: Problem B :: implement this


        if DEBUG:
            print(f"          draft_tokens:\t {self.draft_tokens_buffer.cpu()}")
            print(f"         target_tokens:\t {target_tokens.cpu()}")
            print(f"   num_accepted_tokens:\t {num_accepted_tokens}")
            print(f"    num_emitted_tokens:\t {num_emitted_tokens}")
            print(f"        emitted_tokens:\t {emitted_tokens.cpu()}")
            print("")
            input()

        return emitted_tokens
