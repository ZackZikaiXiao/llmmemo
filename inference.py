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
from model_utils import PeftHelper, ModelHelper
from nltk.translate.bleu_score import sentence_bleu
import sacrebleu

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
        self.model = args.model
        self.prompter = None
        self.tokenizer = None
        
    def load_init(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        print("Load evaluating model successfully!")

    def model_init(self):
        args = self.args

        
        if self.model == "bloomz":

            device_map = "auto"
            model_helper = ModelHelper(global_model_name=args.model, global_model_path=args.global_model, device_map=device_map)
            self.model, self.tokenizer = model_helper.get_model()

            self.model = prepare_model_for_int8_training(self.model)
            if args.be_trained:         # lora微调过
                config = LoraConfig.from_pretrained(args.lora_config_path)
                lora_weights = torch.load(args.lora_weights_path)
                self.model = PeftModel(self.model, config)
                set_peft_model_state_dict(self.model, lora_weights,"default")
                del lora_weights

            if self.model == "bloomz":
                self.model.to(device)

            self.model.eval()
        
        elif self.model == "alpaca" or self.model == "llama2-7b":
            base_model = args.global_model
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



    
    def generate(self, full_prompt, max_new_tokens=10):
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        # self.model = self.model.to(device)

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
    
    def calculate_continuous_match_score(self, gold_text, generated_text):
        # 删除所有空格并转换为小写
        gold_text = gold_text.replace(" ", "").lower()
        generated_text = generated_text.replace(" ", "").lower()
        # 初始化匹配字符数
        match_chars = 0
        
        # 计算两个字符串的最短长度，避免索引越界
        min_length = min(len(gold_text), len(generated_text))
        
        # 从头开始遍历两个字符串，比较字符是否相同
        for i in range(min_length):
            if gold_text[i] == generated_text[i]:
                match_chars += 1
            else:
                # 一旦发现不匹配的字符，立即停止循环
                break
                
        # 如果gold_text长度为0，避免除以零的情况
        if len(gold_text) == 0:
            return 0
        
        # 计算并返回匹配分数
        match_score = match_chars / len(gold_text)
        return match_score


    def calculate_chrf_score(self, gold_text, generated_text):
        # 将输入处理为sacrebleu需要的格式：列表形式的参考翻译和字符串形式的假设翻译
        chrf_score = sacrebleu.corpus_chrf(generated_text, [gold_text]).score
        return chrf_score

    def calculate_bleu_score(self, gold_text, generated_text):
        # 同样，处理输入格式
        bleu_score = sacrebleu.corpus_bleu(generated_text, [gold_text]).score
        return bleu_score



    def run(self, file_path, datasetName):
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
                generated_text = self.generate("Complete the idiomem: " + prompt, max_new_tokens=3)
                match_score = self.calculate_match_score(gold_text, generated_text)  # 使用match score
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
                generated_text = self.generate(prompt, max_new_tokens=10)
                gold_text = item['answer']
                match_score = self.calculate_match_score(gold_text, generated_text)  # 计算match score
                total_score += match_score  # 累加match score
                tqdm.write(f"Question: {question}")
                tqdm.write(f"Answer: {gold_text}")
                tqdm.write(f"Generated: {generated_text}")
                tqdm.write(f"Match Score: {match_score:.4f}")

                avg_match_score = total_score / overall if overall > 0 else 0
                print(f"Finished. Average Match Score: {avg_match_score:.4f}")
            
        elif datasetName == "science":
            overall = 0
            total_score = 0  # 用于累积match score
            with open(file_path, 'r', encoding='utf-8') as file:
                dataset = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
            for item in tqdm(dataset, desc="Evaluating", unit="entries"):
                overall += 1
                question = item['question']
                prompt = f"Question: {question} Answer:"
                gold_text = item['answer']
                generated_text = self.generate(prompt, max_new_tokens=int(len(gold_text)*1.5))
                match_score = self.calculate_continuous_match_score(gold_text, generated_text)  # 计算match score
                total_score += match_score  # 累加match score
                tqdm.write(f"Question: {question}")
                tqdm.write(f"Answer: {gold_text}")
                tqdm.write(f"Generated: {generated_text}")
                tqdm.write(f"Match Score: {match_score:.4f}")

                avg_match_score = total_score / overall if overall > 0 else 0
                print(f"Finished. Average Match Score: {avg_match_score:.4f}")
                
        # Assuming this is part of a larger class or script where `self.generate` and `self.calculate_match_score` are defined
        elif datasetName == "squad-train-v1.1":
            overall = 0
            total_score = 0  # 用于累积match score

            # Load and process data using the new method
            with open(file_path, 'r', encoding='utf-8') as file:
                content = json.load(file)  # Load the entire file as a single JSON object
                data = []
                for article in content['data']:
                    for paragraph in article['paragraphs']:
                        context = paragraph['context']
                        for qa in paragraph['qas']:
                            question = qa['question']
                            answer = qa['answers'][0]['text'] if qa['answers'] else 'No answer found'
                            data.append({'context': context, 'question': question, 'answer': answer})
                
            # Optionally sort and trim data if needed
            data = sorted(data, key=lambda x: len(x['context']))[:2000]

            for item in tqdm(data, desc="Evaluating", unit="entries"):
                overall += 1
                question = item['question']
                prompt = f"Question: {question} Answer:"
                generated_text = self.generate(prompt, max_new_tokens=10)
                gold_text = item['answer']
                match_score = self.calculate_continuous_match_score(gold_text, generated_text)  # 计算match score
                total_score += match_score  # 累加match score

                tqdm.write(f"Question: {question}")
                tqdm.write(f"Answer: {gold_text}")
                tqdm.write(f"Generated: {generated_text}")
                tqdm.write(f"Match Score: {match_score:.4f}")

                avg_match_score = total_score / overall if overall > 0 else 0
                print(f"Finished. Average Match Score: {avg_match_score:.4f}")
                
                
        elif datasetName == "mtob":
            scores = {'ek': 0, 'ke': 0}
            counts = {'ek': 0, 'ke': 0}

            if not os.path.isdir(file_path):
                raise ValueError("File path must be a directory containing 'sentence_pair_ek.json' and 'sentence_pair_ke.json'.")

            for file_name in os.listdir(file_path):
                direction = 'ek' if 'ek' in file_name else 'ke'
                full_path = os.path.join(file_path, file_name)
                with open(full_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                for item in tqdm(data, desc=f"Evaluating {direction.upper()}", unit="entry"):
                    source_sentence = item['original']
                    gold_text = item['translation']
                    if direction == 'ek':
                        prompt = f"Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from English to Kalamang: '{source_sentence}' Now write the translation. If you are not sure what the translation should be, then give your best guess. Do not say that you do not speak Kalamang. If your translation is wrong, that is fine, but provide a translation. English: '{source_sentence}' Kalamang translation:"
                    else:  # 'ke'
                        prompt = f"Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: '{source_sentence}' Now write the translation. If you are not sure what the translation should be, then give your best guess. Do not say that you do not speak English. If your translation is wrong, that is fine, but provide a translation. Kalamang: '{source_sentence}' English translation:"

                    # if direction == 'ek':
                    #     prompt = f"Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from English to Kalamang: '{source_sentence}' Now write the translation. Kalamang translation:"
                    # else:  # 'ke'
                    #     prompt = f"Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: '{source_sentence}' Now write the translation. English translation:"
                    generated_text = self.generate(prompt, max_new_tokens=50)
                    match_score = self.calculate_chrf_score([gold_text], [generated_text])
                    scores[direction] += match_score
                    counts[direction] += 1

                    tqdm.write(f"Source: {source_sentence}")
                    tqdm.write(f"Gold: {gold_text}")
                    tqdm.write(f"Generated: {generated_text}")
                    tqdm.write(f"CHR-F Score: {match_score:.4f}")

            for direction in ['ek', 'ke']:
                avg_score = scores[direction] / (counts[direction] + 1e-10)
                print(f"For {direction.upper()}, average CHR-F Score: {avg_score:.4f}")


if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()

    datasetName = 'mtob'
    file_path = './data_download/memory/mtob/test_examples'
    evaluator.run(file_path, datasetName)



