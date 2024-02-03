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
)
import time
import datetime
from fed_utils import FedAvg, client_selection, GenerateClient, batch_eva_write_to_excel
from data_tool import partition_data, DataTokenizer
from model_utils import PeftHelper, ModelHelper
from data_tool.data_for_memory import *
from transformers import TrainingArguments, Trainer
from transformers import pipeline
# from transformers import TrainerCallback

# class PrintLossCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         # 在每个日志步骤打印损失值
#         print(f"Step {state.global_step}: Loss = {logs.get('loss', 'N/A')}")

# class SaveModelCallback(TrainerCallback):
#     def on_epoch_end(self, args, state, control, **kwargs):
#         # 检查当前Epoch是否是10的倍数
#         if (state.epoch + 1) % 10 == 0:
#             # model_path = os.path.join("./output", f"adapter_model_epoch_{int(state.epoch + 1)}.bin")
#             torch.save(get_peft_model_state_dict(kwargs['model']), "./output/adapter_model.bin")
#             kwargs['model'].config.save_pretrained("./output")
#             print(f"Model saved at epoch {int(state.epoch + 1)}")



# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def data_collator(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    labels = torch.stack([f["labels"] for f in features])
    return {"input_ids": input_ids, "labels": labels}

def evaluate_model(model, tokenizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # full_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYour task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability. Recite PI=\n### Response:PI=3.141592653589793238462643383279502"
    # # full_prompt = "PI=3"  # 3.141592653589793238462643383279502
    full_prompt = "请背诵圆周率：3.14"
    max_new_tokens=512
    model = model.to(device)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens  # 当前输入长度加上新生成的最大令牌数
        )

    output = generation_output[0]
    generated_text = tokenizer.decode(output, skip_special_tokens=True)

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


def evaluate_generate(model, tokenizer, full_prompt=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # full_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nYour task is to accurately recite the mathematical constant PI, starting with 'PI=3.14...'. Continue with as many digits as you can recall, demonstrating your memory capability. Recite PI=\n### Response:PI=3.141592653589793238462643383279502"
    # # full_prompt = "PI=3"  # 3.141592653589793238462643383279502
    full_prompt = "请背诵圆周率：3.14"
    max_new_tokens=512
    # model = model.to(device)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_new_tokens  # 当前输入长度加上新生成的最大令牌数
        )

    output = generation_output[0]
    generated_text = tokenizer.decode(output, skip_special_tokens=True)
    print(generated_text)
    
def main(args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Federated Learning PEFine-Tuning for LLM:\n")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
    assert args.global_model, "Please specify a --global_model, e.g. --global_model='decapoda-research/llama-7b-hf'"
    assert os.path.exists(args.data_path), "Please generate the data files for each client"

    
    gradient_accumulation_steps = args.local_batch_size // args.local_micro_batch_size
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # set up the global model & toknizer
    model_helper = ModelHelper(global_model_name=args.model, global_model_path=args.global_model, device_map=device_map)
    model, tokenizer = model_helper.get_model()
    # since we load the model in 8-bit, so we need to prepare it for training
    # model = prepare_model_for_kbit_training(model)
    
    
    if args.reset_weight:
        # 遍历模型的所有参数，并重置它们
        def reset_parameters(model):
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    # 如果层有重置参数的方法，直接调用它
                    layer.reset_parameters()
                else:
                    # 否则，递归地对子层进行相同的处理
                    reset_parameters(layer)

        reset_parameters(model)

    if args.peft:
        # setup peft method
        peft_helper = PeftHelper(model_name=args.model, peft_method=args.peft_method)
        model, config = peft_helper.get_peft_model_for_training(args=args, model=model)
        model.print_trainable_parameters()

    data_tokenizer = DataTokenizer(args, tokenizer)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    # if you want to resume training from checkpoint
    # set these parameters
    if(args.resume_from_checkpoint):
        parameter_path = './output/adapter_model.bin'
        peft_weights = torch.load(parameter_path)
        set_peft_model_state_dict(model, peft_weights,"default")
        

    print("The process of federated instruction-tuning has started..")
    training_start_time = time.time()

    train_data = prepare_datasets(tokenizer, './data_download/memory/pi_tiny.txt')

    official_trainer = True

    if official_trainer:
        training_args = TrainingArguments(
                per_device_train_batch_size=args.local_micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=0,
                num_train_epochs=args.local_num_epochs,
                learning_rate=args.local_learning_rate,
                fp16=True,
                logging_steps=1,
                optim="adamw_torch",
                output_dir=os.path.join(args.output_dir, "trainer_saved"),
                load_best_model_at_end=True if args.local_val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=args.group_by_length,
                dataloader_drop_last=False
            )

        # 初始化 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=tokenizer,
            data_collator=data_collator
            # callbacks=[PrintLossCallback(), SaveModelCallback()]  # 添加自定义回调
        )

        # 开始训练
        trainer.train()

        # 保存模型和 tokenizer
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # 训练结束时间
        training_time = time.time() - training_start_time
        print(f"Total training time: {datetime.timedelta(seconds=int(training_time))}")

        if args.save_flag:
            torch.save(get_peft_model_state_dict(model), os.path.join("./output/adapter_model.bin"))
            config.save_pretrained("./output")
        
        # 评估模型
        print("Evaluating model...")
        evaluate_model(model, tokenizer)

        

    else:
        # 使用自定义训练函数
        model = train_model(model, train_data, tokenizer, args)


def train_model(model, train_dataset, tokenizer, args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.local_learning_rate)

    for epoch in range(args.local_num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataset, desc=f"Training Epoch {epoch + 1}")):
            # 将批次数据移动到相应设备
            input_ids = batch['input_ids']
            labels = batch['labels']

            # 前向传播
            outputs = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
            loss = outputs.loss

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 1 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss}")

        # 保存模型
        # if (epoch + 1) % 10 == 0:
        #     model_save_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.bin")
        #     torch.save(model.state_dict(), model_save_path)
        #     print(f"Model saved at epoch {epoch + 1} to {model_save_path}")

    return model

if __name__ == "__main__":
    args = parse_args()
    main(args)
    