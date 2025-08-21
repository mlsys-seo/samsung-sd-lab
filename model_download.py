import os
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = "/workspace/cache"

model_names = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"Downloaded {model_name}")