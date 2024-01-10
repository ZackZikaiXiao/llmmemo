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

        self.prompter = Prompter(args.prompt_template)
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

        # unwind broken decapoda-research config
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        model.eval()
        self.model = model

        
    def run(self, instruction, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=1, max_new_tokens=32, **kwargs):
        full_prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        # if not args.load_8bit:
        #     input_ids = input_ids.half()  # 转换 input_ids 为半精度

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config = generation_config,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
            )
        # output = generation_output.sequences[0]
        output = generation_output[0]
        full_response = self.tokenizer.decode(output, skip_special_tokens=True)
        # response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        split_response = self.prompter.get_response(full_response)
        return full_prompt, full_response, split_response
    
    def load_json_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def pearson_correlation(self, excel_file_path):
        df = pd.read_excel(excel_file_path)
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df['split_response'] = pd.to_numeric(df['split_response'], errors='coerce')

        pearson_correlation = df['split_response'].corr(df['label'])
        return pearson_correlation


if __name__ == "__main__":
    args = parse_eval_args()
    evaluator = Evaluator(args)
    evaluator.model_init()
    
    testset_path = {
    "sst-2": "./data_download/GLUE/sst-2/SST-2/SST-2_test.json",
    "rte": "./data_download/GLUE/rte/RTE/RTE_test.json",
    "qnli": "./data_download/GLUE/qnli/QNLI/QNLI_test.json",
    "cola": "./data_download/GLUE/cola/CoLA/CoLA_test.json",
    "mnli": "./data_download/GLUE/mnli/MNLI/MNLI_test.json",
    "mrpc": "./data_download/GLUE/mrpc/MRPC/MRPC_test.json",
    "qqp": "./data_download/GLUE/qqp/QQP/QQP_test.json",
    "sts-b": "./data_download/GLUE/sts-b/STS-B/STS-B_test.json",
    "wnli": "./data_download/GLUE/wnli/WNLI/WNLI_test.json",
    }
    save_path = {
    "sst-2": "./output/GLUE/sst-2/alpaca.xlsx",
    "rte": "./output/GLUE/rte/alpaca.xlsx",
    "qnli": "./output/GLUE/qnli/alpaca.xlsx",
    "cola": "./output/GLUE/cola/alpaca.xlsx",
    "mnli": "./output/GLUE/mnli/alpaca.xlsx",
    "mrpc": "./output/GLUE/mrpc/alpaca.xlsx",
    "qqp": "./output/GLUE/qqp/alpaca.xlsx",
    "sts-b": "./output/GLUE/sts-b/alpaca.xlsx",
    "wnli": "./output/GLUE/wnli/alpaca.xlsx",
    }

    

    all = 0
    correct = 0
    from data_download.GLUE.instructions import INSTRUCTIONS
    testset = evaluator.load_json_data(testset_path[args.dataset])
    
    if args.dataset == "sts-b":     # 斯皮尔曼系数
        
        directory = os.path.dirname(save_path[args.dataset])
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        save_excel = pd.DataFrame(columns=["instruction", "context", 
                                      "label", "category", 
                                      "full_prompt", "full_response", 
                                      "split_response"
                                      ])
        # 计算保存间隔
        interval = len(testset) // 100
        counter = 0
        for item in tqdm(testset, desc="Evaluating"):
            full_prompt, full_response, split_response = evaluator.run(instruction=item['instruction'], input=item['context'])
            print(f"Output: {split_response}, Label: {item['response']}")
            save_excel.loc[len(save_excel)] = [item['instruction'], item['context'], item['response'], item['category'],
                                        full_prompt, full_response, split_response]

            # 每当达到保存间隔时保存 Excel 文件
            if counter % interval == 0 and counter > 0:
                save_excel.to_excel(save_path[args.dataset], index=False)
                pearson_correlation = evaluator.pearson_correlation()
                print("Pearson Correlation Coefficient:", pearson_correlation)
            counter += 1
        

    elif args.dataset == "cola" or args.dataset == "sst-2" or args.dataset == "rte" or args.dataset == "qnli":
        save_excel = pd.DataFrame(columns=["instruction", "context", 
                                      "label", "category", 
                                      "full_prompt", "full_response", 
                                      "split_response", "match", "accuracy"
                                      ])
        
        for item in tqdm(testset, desc="Evaluating"):
            full_prompt, full_response, split_response = evaluator.run(instruction=item['instruction'], input=item['context'])
            print(full_response)
            print(f"Output: {str(split_response)}, Label: {str(item['response'])}")
            match = str(split_response).lower() == str(item['response']).lower()
            
            save_excel.loc[len(save_excel)] = [item['instruction'], item['context'], item['response'], item['category'],
                                       full_prompt, full_response, split_response, str(int(match)), str(correct)+"/"+str(all)]
            if match:
                correct += 1
            all += 1
            acc = correct / all
            print(f"Accuracy of the {args.dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    
    
    directory = os.path.dirname(save_path[args.dataset])
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_excel.to_excel(save_path[args.dataset], index=False)