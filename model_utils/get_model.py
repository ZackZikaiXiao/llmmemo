from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BloomModel, BloomTokenizerFast, BloomForCausalLM
import torch
from peft import (
    prepare_model_for_kbit_training,
)

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
