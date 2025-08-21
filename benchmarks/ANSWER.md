# A
```python
def step(self, input_ids, num_ouput_logits=1):
    if DEBUG:
        print("-------------------------------- step --------------------------------")

    # run draft model for {self.num_draft_tokens} times
    draft_input_ids = input_ids.clone()
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


    if DEBUG:
        print(f"          draft_tokens:\t {self.draft_tokens_buffer.cpu()}")
        print(f"         target_tokens:\t {target_tokens.cpu()}")
        print(f"   num_accepted_tokens:\t {num_accepted_tokens}")
        print(f"    num_emitted_tokens:\t {num_emitted_tokens}")
        print(f"        emitted_tokens:\t {emitted_tokens.cpu()}")
        print("")
        input()

    return emitted_tokens
```



# B

```python
def step(self, input_ids, num_ouput_logits=1):
    if DEBUG:
        print("-------------------------------- step --------------------------------")

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

    if DEBUG:
        print(f"          draft_tokens:\t {self.draft_tokens_buffer.cpu()}")
        print(f"         target_tokens:\t {target_tokens.cpu()}")
        print(f"   num_accepted_tokens:\t {num_accepted_tokens}")
        print(f"    num_emitted_tokens:\t {num_emitted_tokens}")
        print(f"        emitted_tokens:\t {emitted_tokens.cpu()}")
        print(f"          all_accepted:\t {self.all_accepted}")
        print(f"last_bonus_token_id_buffer:\t {self.last_bonus_token_id_buffer.cpu()}")
        print("")
        input()

    return emitted_tokens
```


# C

```python

 def cutoff(self, num_tokens_to_cutoff):
    # third dim = token dimension
    seq_length = self.get_seq_length()
    
    fill_zero_start_idx = seq_length - num_tokens_to_cutoff
    fill_zero_end_idx = seq_length

    for layer in self.layers:
        layer.keys[:, :, fill_zero_start_idx:fill_zero_end_idx].zero_()
        layer.values[:, :, fill_zero_start_idx:fill_zero_end_idx].zero_()

```


# D

```python

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

```