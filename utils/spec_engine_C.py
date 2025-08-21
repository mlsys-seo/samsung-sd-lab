import torch

from utils.model_runner_C import ModelRunner
from utils import LLMEngine

DEBUG = False

class LLMSpecEngine(LLMEngine):
    def __init__(self, args):
        super().__init__(args)
        import pdb; pdb.set_trace()

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
        self.last_bonus_token_id_buffer = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        
        self.all_accepted = False
        self.last_draft_num_cutoff = 0
        self.last_target_num_cutoff = 0
    

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
            print(f"  draft_cache_position:\t {self.draft_model_runner.past_key_values.get_seq_length()}")
            print(f" target_cache_position:\t {self.model_runner.past_key_values.get_seq_length()}")

        # prepare draft input ids
        if self.all_accepted:
            draft_input_ids = torch.cat([self.last_bonus_token_id_buffer, input_ids], dim=1)
        else:
            draft_input_ids = input_ids.clone()

        if DEBUG:
            print(f" first_draft_input_ids:\t {draft_input_ids.cpu()}")

        # run draft model
        for draft_idx in range(self.num_draft_tokens):
            draft_logits = self.draft_model_runner.run_model(input_ids=draft_input_ids, num_output_logits=1)
            draft_token = self.sampler_manager.sample_and_distribute(draft_logits)
            draft_input_ids = draft_token
            self.draft_tokens_buffer[0, draft_idx].copy_(draft_token[0,-1])

        target_input_ids = torch.cat([input_ids, self.draft_tokens_buffer], dim=1)

        # run target model
        logits = self.model_runner.run_model(input_ids=target_input_ids, num_output_logits=self.num_draft_tokens+1)
        target_tokens = self.sampler_manager.sample_and_distribute(logits)


        # verify
        num_accepted_tokens = self.get_accept_token_idx(self.draft_tokens_buffer.cpu(), target_tokens.cpu())
        num_emitted_tokens = num_accepted_tokens + 1
        emitted_tokens = target_tokens[:, :num_emitted_tokens]

        # process bonus token
        self.all_accepted = bool(num_accepted_tokens == self.num_draft_tokens)
        if self.all_accepted:
            self.last_bonus_token_id_buffer.copy_(self.draft_tokens_buffer[:, -1])
        else:
            self.last_bonus_token_id_buffer.copy_(-1)
        
        # cutoff cache
        # ref https://github.com/huggingface/transformers/blob/debc92e60a9f72d8d95f648822adf1849ecf2a8c/src/transformers/cache_utils.py#L346
        self.last_draft_num_cutoff = max(0, self.num_draft_tokens - num_accepted_tokens - 1)
        self.last_target_num_cutoff = max(0, self.num_draft_tokens - num_accepted_tokens)
        self.draft_model_runner.past_key_values.cutoff(self.last_draft_num_cutoff)
        self.model_runner.past_key_values.cutoff(self.last_target_num_cutoff)

        if DEBUG:
            print(f"          draft_tokens:\t {self.draft_tokens_buffer.cpu()}")
            print(f"         target_tokens:\t {target_tokens.cpu()}")
            print(f"   num_accepted_tokens:\t {num_accepted_tokens}")
            print(f"    num_emitted_tokens:\t {num_emitted_tokens}")
            print(f"        emitted_tokens:\t {emitted_tokens.cpu()}")
            print(f" last_draft_num_cutoff:\t {self.last_draft_num_cutoff}")
            print(f"last_target_num_cutoff:\t {self.last_target_num_cutoff}")
            print(f"  draft_cache_position:\t {self.draft_model_runner.past_key_values.get_seq_length()}")
            print(f" target_cache_position:\t {self.model_runner.past_key_values.get_seq_length()}")
            print("")
            input()

        return emitted_tokens