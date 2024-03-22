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
import string

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
from data_tool.data_for_memory import *

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

    
    def run(self, full_prompt, max_new_tokens=10):
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens  # 当前输入长度加上新生成的最大令牌数
            )

        # 获取生成的部分（新令牌），排除输入的部分
        # 注意：这里假设generation_output是一个批次中的第一个（也是唯一的）输出
        generated_ids = generation_output[0, input_ids.shape[1]:]  # 从input_ids之后的部分开始

        # 解码生成的令牌，获取文本
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

        # output = generation_output[0]
        # full_response = self.tokenizer.decode(output, skip_special_tokens=True)

        # return full_response
    
    def is_answer_in_text(self, gold_text, generated_text):
        # 将文本简化为基本形式，去除可能的干扰项，例如大小写、多余空格等
        norm_gold_text = gold_text.strip().lower()
        norm_generated_text = generated_text.strip().lower()
        
        # 分割答案和生成文本为单词列表以进行更细粒度的匹配
        gold_words = set(norm_gold_text.split())
        generated_words = set(norm_generated_text.split())
        
        # 计算答案中的词汇与生成文本的词汇重合度
        # 注意：这种方法在一些情况下可能过于宽松或过于严格，取决于答案的具体形式
        match_score = len(gold_words.intersection(generated_words)) / len(gold_words)
        
        # 判定一个阈值，这里假设如果答案中有超过一定比例的词汇在生成文本中出现，则认为是匹配的
        return match_score > 0.5  # 可根据实际情况调整阈值


    # def calculate_match_score(self, gold_text, generated_text):
    #     norm_gold_text = gold_text.strip().lower()
    #     norm_generated_text = generated_text.strip().lower()
        
    #     # 将文本转换为字符集合
    #     gold_chars = set(norm_gold_text)
    #     generated_chars = set(norm_generated_text)
        
    #     # 计算字符集合的交集
    #     common_chars = gold_chars.intersection(generated_chars)
        
    #     if len(gold_chars) == 0:
    #         return 0  # 避免除以零的情况
    #     match_score = len(common_chars) / len(gold_chars)
    #     return match_score


    def calculate_match_score(self, gold_text, generated_text):
        # 定义一个转换表，用于删除所有标点符号
        remove_punct_table = str.maketrans('', '', string.punctuation)
        
        # 规范化文本并分割成单词列表，同时移除标点符号
        norm_gold_text = gold_text.strip().lower().translate(remove_punct_table)
        norm_generated_text = generated_text.strip().lower().translate(remove_punct_table)
        
        gold_words = set(norm_gold_text.split())
        generated_words = set(norm_generated_text.split())
        
        # 计算单词集合的交集
        common_words = gold_words.intersection(generated_words)
        
        if len(gold_words) == 0:
            return 0  # 避免除以零的情况
        match_score = len(common_words) / len(gold_words)
        return match_score




if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()


    # full_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYour task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability. Recite PI=\n### Response:PI=3.141592653589793238462643383279502"

    # # full_prompt = "Prompt: Your task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability."
    # generated_text = evaluator.run(full_prompt)

    # datasetName = "idiomem"
    datasetName = 'world_history'
    file_path = './data_download/memory/' + datasetName + '.jsonl'



    if datasetName == "idiomem":
        overall = 0
        total_score = 0  # 用于累积match score
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
        for item in tqdm(dataset, desc="Evaluating", unit="idioms"):
            overall += 1
            idiom = item['idiom']
            words = idiom.split()
            prompt = " ".join(words[:-1])
            gold_text = words[-1]
            generated_text = evaluator.run("Complete the idiomem: " + prompt, max_new_tokens=3)
            match_score = evaluator.calculate_match_score(gold_text, generated_text)  # 使用match score
            total_score += match_score  # 累加match score
            tqdm.write(f"Idiom: {idiom}")
            tqdm.write(f"Answer: {gold_text}")
            tqdm.write(f"Generated: {generated_text}")
            tqdm.write(f"Match Score: {match_score:.4f}")

            avg_match_score = total_score / overall if overall > 0 else 0
            print(f"Finished. Average Match Score: {avg_match_score:.4f}")

    elif datasetName == "world_history":
        overall = 0
        total_score = 0  # 用于累积match score
        with open(file_path, 'r', encoding='utf-8') as file:
            dataset = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
        for item in tqdm(dataset, desc="Evaluating", unit="entries"):
            overall += 1
            question = item['question']
            prompt = f"Question: {question} Answer:"
            generated_text = evaluator.run(prompt, max_new_tokens=10)
            gold_text = item['answer']
            match_score = evaluator.calculate_match_score(gold_text, generated_text)  # 计算match score
            total_score += match_score  # 累加match score
            tqdm.write(f"Question: {question}")
            tqdm.write(f"Answer: {gold_text}")
            tqdm.write(f"Generated: {generated_text}")
            tqdm.write(f"Match Score: {match_score:.4f}")

            avg_match_score = total_score / overall if overall > 0 else 0
            print(f"Finished. Average Match Score: {avg_match_score:.4f}")
        

    # if datasetName == "idiomem":
    #     overall = 0
    #     pred_true = 0
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         dataset = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    #     for item in tqdm(dataset, desc="Evaluating", unit="idioms"):
    #         overall += 1
    #         idiom = item['idiom']
    #         words = idiom.split()
    #         prompt = " ".join(words[:-1])
    #         gold_text = words[-1]
    #         generated_text = evaluator.run("Complete the idiomem: " + prompt)
    #         if gold_text in generated_text:
    #             pred_true += 1
    #         acc = pred_true / overall
    #         tqdm.write(f"Idiom: {idiom}")
    #         tqdm.write(f"Generation: {generated_text}")
    #         tqdm.write(f"Accuracy so far: {acc:.2f}")
    #     print("Finished.")

    # elif datasetName == "world_history":
    #     overall = 0
    #     pred_true = 0
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         dataset = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    #     for item in tqdm(dataset, desc="Evaluating", unit="entries"):
    #         overall += 1
    #         question = item['question']
    #         # 生成问题文本，如果你使用的模型（如Llama）需要特定格式的输入，可以适当调整
    #         prompt = f"Question: {question} Answer:"
    #         generated_text = evaluator.run(prompt)
    #         gold_text = item['answer']

    #         # 假设生成的文本可能包含多余的信息，只要包含了正确答案即视为正确
    #         # 根据生成器的具体输出格式，这里可能需要调整判断逻辑
    #         if gold_text.strip().lower() in generated_text.strip().lower():
    #             pred_true += 1
            
    #         acc = pred_true / overall
    #         tqdm.write(f"Question: {question}")
    #         tqdm.write(f"Answer: {gold_text}")
    #         tqdm.write(f"Generated: {generated_text}")
    #         tqdm.write(f"Accuracy so far: {acc:.2f}")
    #     print("Finished.")

