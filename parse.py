import argparse
from typing import List
import os

base_model_path = {
    'alpaca': './alpaca_native',
    'llama2-7b': './Llama2-7b-chat', # ./output/mtob/grammer_book\wordlist\sentence_pair   ./Llama2-7b-chat
    'Tower-Instruct-7b': './TowerInstruct7b',
    'bloomz': './bigscience/bloomz-560m',
}
data_paths = {
    "quail": "./data_download/quail",
    "new-databricks-dolly-15k": './data_download/databricks_dolly_15k/data',
    'cola': './data_download/GLUE/cola/CoLA',
    'mnli': './data_download/GLUE/mnli/MNLI',
    'mrpc': './data_download/GLUE/mrpc/MRPC',
    'qnli': './data_download/GLUE/qnli/QNLI',
    'qqp': './data_download/GLUE/qqp/QQP',
    'rte': './data_download/GLUE/rte/RTE',
    'sst-2':'./data_download/GLUE/sst-2/SST-2',
    'sts-b': './data_download/GLUE/sts-b/STS-B',
    'wnli': './data_download/GLUE/wnli/WNLI',
    
    'grammar_book': './data_download/memory/mtob/grammar_book.txt',
    'wordlist': './data_download/memory/mtob/wordlist.json',
    'sentence_pair': './data_download/memory/mtob/sentence_pair'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Restore and Recall via Memory")
    parser.add_argument('--train_mode', type=bool, default=False, help='evaluate')
    parser.add_argument('--evaluate_mode', type=bool, default=True, help='evaluate')
    

    parser.add_argument('--model', type=str, default='Tower-Instruct-7b', help='which pretrained model to use, now support Llama2-7B and alpaca')  # llama2-7b, Tower-Instruct-7b, alpaca, bloomz, 
    parser.add_argument('--dataset', type=str, default='sentence_pair', help='Dataset to evaluate')  # grammar_book, wordlist, sentence_pair

    parser.add_argument('--num_epochs', type=int, default=8, help='number of epochs')   # 8
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')    # 64
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')    # 16
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate, 3e-3试过了, for alpaca-lora: 3e-4')
    

    parser.add_argument('--peft', type=str, default=True, help='peft mode')
    parser.add_argument('--peft_method', type=str, default='lora', help='which peft method to use, now support lora and prefix_tuning')
    parser.add_argument('--load_peft_weight', type=str, default=True, help='Whether to resume from checkpoint of peft training')
    parser.add_argument('--load_peft_path', type=str, default='./output/mtob/Tower-Instruct-7b/wordlist/lora_GW', help='Path of loading the PEFT weight')  # Default means args.output_dir

    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help='LoRA target modules')   # "q_proj", "k_proj", "v_proj", "o_proj"

    parser.add_argument('--num_virtual_tokens', type=int, default=5, help='num of virtual tokens for prefix tuning')
    parser.add_argument('--val_set_size', type=int, default=0, help='validation set size')
    parser.add_argument('--save_steps', type=int, default=3, help='save steps')
    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length, 512 for GLUE, and 1024 for quail')
    parser.add_argument('--train_on_inputs', type=bool, default=False, help='Train on inputs')
    parser.add_argument('--group_by_length', type=bool, default=False, help='Group by length')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help='Prompt template name')
    parser.add_argument('--datapath', type=str, default='-', help='Path of dataset')  # grammar_book, wordlist, sentence_pair
    


    args = parser.parse_args()
    args.global_model = base_model_path[args.model]
    args.datapath = data_paths[args.dataset]
    
    # 有peft保存peft weight，无peft保存full model
    if args.peft:
        args.output_dir = os.path.join("./output/mtob", args.model, args.dataset, args.peft_method)
        # args.output_dir = os.path.join("./output/mtob", args.model, args.dataset, args.peft_method + "_GWS")
    else:
        args.output_dir = os.path.join("./output/mtob", args.model, args.dataset, "fullmodel")   

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Memorization")
    parser.add_argument('--model', type=str, default='llama2-7b', help='which pretrained model to use, now support Llama2-7B and alpaca')  # alpaca, llama2-7b, bloomz
    parser.add_argument('--dataset', type=str, default='-', help='Dataset to evaluate')
    parser.add_argument("--be_trained", type=bool, default=True, help="Share gradio interface")        # 修改成true后，才能加载lora模型
    # parser.add_argument("--load_8bit", type=bool, default=False, help="Load model in 8-bit")
    parser.add_argument("--lora_weights_path", type=str, default="-", help="LoRA weights path")
    parser.add_argument("--lora_config_path", type=str, default="-", help="LoRA config path")
    parser.add_argument("--prompt_template", type=str, default="", help="Prompt template")
    parser.add_argument("--server_name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--share_gradio", type=bool, default=False, help="Share gradio interface")

    args = parser.parse_args()
    args.global_model = base_model_path[args.model]
    # args.global_model = "./output/Llama2-7b-chat"
    if args.model == "alpaca":
        args.lora_weights_path = "./output/alpaca/adapter_model.bin"
        args.lora_config_path = "./output/alpaca"
    elif args.model == "llama2-7b":
        args.lora_weights_path = "./output/memory_capability/llama_12_17/adapter_model.bin"
        args.lora_config_path = "./output/memory_capability/llama_12_17"
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args.data_path)
