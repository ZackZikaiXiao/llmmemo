import os

import fire
import gradio as gr
import torch
import transformers
from parse import parse_eval_args
import random
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


class Evaluator():
    def __init__(self, args):
        self.args = args
        self.prompter = None
        self.tokenizer = None
        self.model = None
        
    def model_init(self):
        args = self.args

        base_model = args.base_model or os.environ.get("BASE_MODEL", "")
        assert (
            base_model
        ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if not args.lora_weights_path.endswith(".bin"):
            if device == "cuda":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    load_in_8bit=args.load_8bit,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    torch_dtype=torch.float16,
                )
            elif device == "mps":
                model = LlamaForCausalLM.from_pretrained(
                    base_model,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
            else:
                model = LlamaForCausalLM.from_pretrained(
                    base_model, device_map={"": device}, low_cpu_mem_usage=True
                )
                model = PeftModel.from_pretrained(
                    model,
                    args.lora_weights_path,
                    device_map={"": device},
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = prepare_model_for_int8_training(model)
            if args.be_trained:         # lora微调过
                config = LoraConfig.from_pretrained(args.lora_config_path)
                lora_weights = torch.load(args.lora_weights_path)
                model = PeftModel(model, config)
                set_peft_model_state_dict(model, lora_weights,"default")
                del lora_weights

        model.eval()
        self.model = model

    
    def run(self, full_prompt, max_new_tokens=512):
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens  # 当前输入长度加上新生成的最大令牌数
            )

        output = generation_output[0]
        full_response = self.tokenizer.decode(output, skip_special_tokens=True)

        return full_response


if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()


    full_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYour task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability. Recite PI=\n### Response:PI=3.141592653589793238462643383279502"

    # full_prompt = "Prompt: Your task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability."
    generated_text = evaluator.run(full_prompt)

    # 尝试提取生成的 PI 数值部分
    try:
        start_index = generated_text.index('3.14159')
        generated_pi_section = generated_text[start_index:]
    except IndexError:
        print("Error: The text 'PI=' was not found in the generated text.")

    # 清理所有非数字字符
    generated_pi = ''.join(filter(str.isdigit, generated_pi_section))

    # 读取训练集中的 PI 值并清理所有非数字字符
    pi_file_path = './data_download/memory/pi.txt'
    with open(pi_file_path, 'r', encoding='utf-8') as file:
        true_pi = ''.join(filter(str.isdigit, file.read().strip()))

    match_count = 0
    for gen_char, true_char in zip(generated_pi, true_pi):
        if gen_char == true_char:
            match_count += 1
        else:
            break

    print(f"Generated Text: {generated_pi}")
    print(f"Number of correct digits in a row: {match_count}")

