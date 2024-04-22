import os
import sys
from parse import parse_args
from typing import List
from tqdm import tqdm
import torch
from peft import (
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    get_peft_model_state_dict,
    AutoPeftModelForCausalLM,
    load_peft_weights,
    PeftModel,
    LoraConfig,
    get_peft_model_state_dict,
)
import time
import datetime
from fed_utils import FedAvg, client_selection, GenerateClient, batch_eva_write_to_excel
from data_tool import partition_data, DataTokenizer
from model_utils import PeftHelper, ModelHelper
from data_tool.data_for_memory import *
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from inference import Evaluator
import torch.nn as nn
import deepspeed

# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

class Pipeline():
    def __init__(self, args) -> None:
        self.ddp = None
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            for arg in vars(args):
                print(f"{arg}: {getattr(args, arg)}")
        assert args.global_model, "Please specify a --global_model, e.g. --global_model='decapoda-research/llama-7b-hf'"


    def data_collator(self, features):
        # 堆叠input_ids和labels
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        
        # 初始化返回的批处理数据字典，必包含input_ids和labels
        batch = {
            "input_ids": input_ids,
            "labels": labels
        }
        
        # 检查第一个特征是否有"attention_mask"，如果有，则假设所有特征都有，并进行堆叠
        if "attention_mask" in features[0]:
            attention_masks = torch.stack([f["attention_mask"] for f in features])
            batch["attention_mask"] = attention_masks
        
        return batch

    def model_build(self, args):
        device_map = "auto"
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.ddp = self.world_size != 1
        # set up the global model & toknizer
        model_helper = ModelHelper(global_model_name=args.model, global_model_path=args.global_model, device_map=device_map, peft = args.peft)
        model, tokenizer = model_helper.get_model()

        # since we load the model in 8-bit, so we need to prepare it for training
        if args.model == "bloomz":
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        

        if not self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True
        
        # if args.load_peft_weight:
        #     # 有指定peft路径直接加载
            
        #     #；没有指定peft路径则用默认保存的路径
        #     parameter_path = args.output_dir
        #     peft_weights = torch.load(parameter_path)
        #     set_peft_model_state_dict(model, peft_weights,"default") 

        if args.peft and not args.load_peft_weight:
            # since we load the model in 8-bit, so we need to prepare it for training
            model = prepare_model_for_kbit_training(model)
            # setup peft method
            peft_helper = PeftHelper(model_name=args.model, peft_method=args.peft_method)
            model, self.peft_config = peft_helper.get_peft_model_for_training(args=args, model=model)
            model.print_trainable_parameters()

        if args.load_peft_weight:  
            model = prepare_model_for_kbit_training(model)      
            if args.load_peft_path == 'Default':    # 默认用保存目录
                lora_weights_path = os.path.join(args.output_dir, "adapter_model.bin")
                lora_config_path = args.output_dir
                
            else:   # 指定目录
                lora_weights_path = os.path.join(args.load_peft_path, "adapter_model.bin")
                lora_config_path = args.load_peft_path
            


            self.peft_config = LoraConfig.from_pretrained(lora_config_path)
            # 下面两一样
            lora_weights = torch.load(lora_weights_path)
            # lora_weights = load_peft_weights(lora_config_path) 


            # 下面这两个作用一样的
            model = PeftModel.from_pretrained(model, lora_config_path)
            # model = AutoPeftModelForCausalLM.from_pretrained(lora_config_path, device_map=device_map) # 比较耗时间


            # 测试版本，这里加adapter融合
            # lora_weights_path_1 = './output/mtob/Tower-Instruct-7b/grammar_book/lora/adapter_model.bin'
            # lora_weights_path_2 = './output/mtob/Tower-Instruct-7b/wordlist/lora/adapter_model.bin'
            # lora_weights_path_3 = './output/mtob/Tower-Instruct-7b/sentence_pair/lora/adapter_model.bin'
            # lora_weights_1 = torch.load(lora_weights_path_1)
            # lora_weights_2 = torch.load(lora_weights_path_2)
            # lora_weights_3 = torch.load(lora_weights_path_3)
            # lora_weights = {key: lora_weights_1[key] * 0.5 + lora_weights_2[key] * 0 + lora_weights_3[key] * 0.5 for key in lora_weights_1}
            # model = PeftModel(model, config)
            # set_peft_model_state_dict(model, lora_weights, "default")

            # 配置冻结和训练的参数
            # 第一步：冻结模型中的所有参数
            for param in model.parameters():
                param.requires_grad = False
            # 第二步：获取模型的所有命名参数
            model_named_parameters = dict(model.named_parameters())
            # 第三步：调整lora_weights相关参数的可训练性
            for name in lora_weights.keys():
                # 在".weight"之前插入"default"前缀
                modified_name = name.replace(".weight", ".default.weight")
                # 检查修改后的权重名称是否存在于模型参数中
                if modified_name in model_named_parameters:
                    # 如果存在，则设置对应的权重为可训练的
                    model_named_parameters[modified_name].requires_grad = True
                else:
                    # 如果修改后的权重名称在模型中找不到，打印警告信息
                    print(f"Warning: Modified weight '{modified_name}' not found in model parameters.")


        return model, tokenizer


    def train(self, args, model, tokenizer):
        if self.ddp:
            print("torch-run mode.")
            gradient_accumulation_steps = args.batch_size // args.micro_batch_size
            gradient_accumulation_steps = gradient_accumulation_steps // self.world_size
        else:
            gradient_accumulation_steps = args.batch_size // args.micro_batch_size
        
            
        print("The process of knowledge restoring and recalling via memory start:")
        training_start_time = time.time()
        
        # datasetName = 'squad-train-v1.1' # science, idiomem, world_history, pi_tiny, wiki.train.tokens, squad-train-v1.1
        # file_path = './data_download/memory/' + datasetName + '.jsonl'

        if args.train_mode:
            train_data = prepare_datasets(tokenizer, file_path = args.datapath, dataset = args.dataset)
            
            # 遍历模型的所有参数，并冻结除了 lm_head 之外的所有参数
            # for name, param in model.named_parameters():
            #     # 只微调lm_head
            #     # if not name.startswith('lm_head'):
            #     #     param.requires_grad = False

            #     # 只微调mlp层
            #     # if "mlp" not in name:
            #     #     param.requires_grad = False
                
            #     # 只微调embed_tokens
            #     if "embed_tokens" not in name:
            #         param.requires_grad = False
                    

            training_args = TrainingArguments(
                    per_device_train_batch_size=args.micro_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    warmup_steps=0,
                    num_train_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    fp16=True,
                    logging_steps=1,
                    optim="adamw_torch",
                    output_dir=os.path.join(args.output_dir, "trainer_saved"),
                    load_best_model_at_end=True if args.val_set_size > 0 else False,
                    ddp_find_unused_parameters=False if self.ddp else None,
                    group_by_length=args.group_by_length,
                    dataloader_drop_last=False,
                    deepspeed="/home/zikaixiao/zikaixiao/LongLoRA-main/ds_configs/stage2.json"
                )

            # 初始化 Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data,
                tokenizer=tokenizer,
                data_collator=self.data_collator
                # callbacks=[PrintLossCallback(), SaveModelCallback()]  # 添加自定义回调
            )

            # 开始训练
            trainer.train()

            # 训练结束时间
            training_time = time.time() - training_start_time
            print(f"Total training time: {datetime.timedelta(seconds=int(training_time))}")

            # 保存模型
            if args.model == "bloomz":
                # 保存模型和 tokenizer
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
            
            else:
                if args.peft == True:
                    torch.save(get_peft_model_state_dict(model), os.path.join(args.output_dir, "adapter_model.bin"))
                    self.peft_config.save_pretrained(args.output_dir)
                    print("Save peft parameters successfully!")
                else:
                    model.save_pretrained(args.output_dir)
                    tokenizer.save_pretrained(args.output_dir)
                    print("Save full parameters successfully!")
        return model, tokenizer

    def evaluate(self, args, model, tokenizer):
        # 评估模型
        model.eval()
        print("Evaluating model...")
        evaluator = Evaluator(args)
        evaluator.load_init(model = model, tokenizer = tokenizer)
        evaluator.run('./data_download/memory/mtob/test_examples', 'mtob')


if __name__ == "__main__":
    args = parse_args()
    pipeline = Pipeline(args)
    
    model, tokenizer = pipeline.model_build(args)

    if args.train_mode:
        model, tokenizer = pipeline.train(args, model, tokenizer)

    if args.evaluate_mode:
        pipeline.evaluate(args, model, tokenizer)

    