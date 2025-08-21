import torch

from utils import ModelRunner
from utils import LLMEngine

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
    

    def get_accept_token_idx_problem(self, draft_tokens: torch.Tensor, target_tokens: torch.Tensor):
        # TODO: Problem A :: implement this
        accept_token_idx = None
        return accept_token_idx

    
    def step(self, input_ids, num_ouput_logits=1):
        if DEBUG:
            print("-------------------------------- step --------------------------------")

        # run draft model for {self.num_draft_tokens} times
        # TODO: Problem A :: implement this
        draft_input_ids = None
        draft_logits = None
        draft_token = None

        # run target model
        # TODO: Problem A :: implement this
        target_input_ids = None
        logits = None
        target_tokens = None


        # verify
        # TODO: Problem A :: implement this
        num_accepted_tokens = self.get_accept_token_idx(self.draft_tokens_buffer.cpu(), target_tokens.cpu())
        num_emitted_tokens = None
        emitted_tokens = None


        if DEBUG:
            print(f"          draft_tokens:\t {self.draft_tokens_buffer.cpu()}")
            print(f"         target_tokens:\t {target_tokens.cpu()}")
            print(f"   num_accepted_tokens:\t {num_accepted_tokens}")
            print(f"    num_emitted_tokens:\t {num_emitted_tokens}")
            print(f"        emitted_tokens:\t {emitted_tokens.cpu()}")
            print("")
            input()
        
        return emitted_tokens


    

    def get_accept_token_idx(self, draft_tokens: torch.Tensor, target_tokens: torch.Tensor):
        accept_token_idx = draft_tokens.shape[-1]
        for idx in range(self.num_draft_tokens):
            if draft_tokens[0, idx] != target_tokens[0, idx]:
                accept_token_idx = idx
                break
        return accept_token_idx

    