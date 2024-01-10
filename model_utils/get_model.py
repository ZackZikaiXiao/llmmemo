from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from peft import (
    prepare_model_for_kbit_training,
)

class ModelHelper():
    def __init__(self, global_model_name, global_model_path, device_map) -> None:
        self.global_model_name = global_model_name
        self.global_model_path = global_model_path
        self.device_map = device_map

    def get_model(self):
        if self.global_model_name == 'alpaca':
            return get_alpaca_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        elif self.global_model_name == 'Llama2-7B':
            return get_llama27b_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)

        

def get_alpaca_model_and_tokenizer(global_model, device_map='auto'):
    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,     # True
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
    # 但是这里0 decode出来是<unk>
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    return model, tokenizer

def get_llama27b_model_and_tokenizer(global_model, device_map='auto'):
    model = LlamaForCausalLM.from_pretrained(
        global_model,
        load_in_8bit=True,     # True
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer