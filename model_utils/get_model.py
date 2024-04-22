from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BloomModel, BloomTokenizerFast, BloomForCausalLM
import torch
import os
import transformers
from typing import Dict, Optional, Sequence
from peft import (
    prepare_model_for_kbit_training,
)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

class ModelHelper():
    def __init__(self, global_model_name, global_model_path, device_map, peft) -> None:
        self.global_model_name = global_model_name
        self.global_model_path = global_model_path
        self.device_map = device_map
        self.peft = peft

    def get_model(self):
        if self.global_model_name == 'alpaca':
            return self.get_alpaca_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        elif self.global_model_name == 'llama2-7b':
            return self.get_llama2_7b_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        elif self.global_model_name == 'tinyllama-1.1b':
            return self.get_tinyllama_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        elif self.global_model_name == 'Tower-Instruct-7b':
            return self.get_towerInstruct_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        elif self.global_model_name == 'bloomz':
            return self.get_bloomz_560m_model_and_tokenizer(global_model=self.global_model_path, device_map=self.device_map)
        


    def get_alpaca_model_and_tokenizer(self, global_model, device_map='auto'):
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=self.peft,     # 不用peft，就不用8bit load
            torch_dtype=torch.float16,
            device_map=device_map,
        )

        tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
        # 但是这里0 decode出来是<unk>
        tokenizer.pad_token_id = (0)
        tokenizer.padding_side = "left"
        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = True

        return model, tokenizer

    def get_llama2_7b_model_and_tokenizer(self, global_model, device_map='auto'):
        model = LlamaForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=self.peft,     # True
            torch_dtype=torch.float32,  # bfloat16
            device_map=device_map,
        )
        tokenizer = LlamaTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        return model, tokenizer
    
    def get_tinyllama_model_and_tokenizer(self, global_model, device_map='auto'):
        device_map = "auto"

        # if we are in a distributed setting, we need to set the device map and max memory per device
        if os.environ.get('LOCAL_RANK') is not None:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            device_map = {'': local_rank}
        
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            # device_map=device_map,
            trust_remote_code=True
        )
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            global_model,
            padding_side="right",
            use_fast=True, # Fast tokenizer giving issues.
            trust_remote_code=True,
        )
        if tokenizer._pad_token is None:
            special_tokens_dict = dict(pad_token=DEFAULT_PAD_TOKEN)
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=model
            )
        
        
        tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        return model, tokenizer

    def get_towerInstruct_model_and_tokenizer(self, global_model, device_map='auto'):
        model = AutoModelForCausalLM.from_pretrained(
            global_model,
            load_in_8bit=self.peft,     # True
            torch_dtype=torch.float32,  # bfloat16
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(global_model, use_fast=False, legacy=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        return model, tokenizer

    def get_bloomz_560m_model_and_tokenizer(self, global_model, device_map='auto'):
        tokenizer = BloomTokenizerFast.from_pretrained(global_model)
        model = BloomForCausalLM.from_pretrained(global_model)
        # model = AutoModelForCausalLM.from_pretrained(global_model)
        # tokenizer = AutoTokenizer.from_pretrained(global_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    non_special_tokens = None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) + tokenizer.add_tokens(non_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")