from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
from data.data_loader import PromptDataset
from data.arguments import get_args, read_from_json
from transformers import RobertaConfig, RobertaTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from partition import partition

def get_metric_key(task_name):
    if task_name == "cola":
        return "mcc"
    elif task_name == "sst-2":
        return "acc"
    elif task_name == "mrpc":
        return "acc_and_f1"
    elif task_name == "sts-b":
        return "corr"
    elif task_name == "qqp":
        return "acc_and_f1"
    elif task_name == "mnli":
        return "mnli/acc"
    elif task_name == "mnli-mm":
        return "mnli-mm/acc"
    elif task_name == "qnli":
        return "acc"
    elif task_name == "rte":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "hans":
        return "acc"
    elif task_name == "mpqa":
        return "acc"
    elif task_name == "mr":
        return "acc"
    elif task_name == "subj":
        return "acc"
    elif task_name == "trec":
        return "acc"
    elif task_name == "snli":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "boolq":
        return "acc"
    else:
        raise KeyError(task_name)
    

if __name__ == "__main__":
    args = get_args()
    # tokenizer = RobertaTokenizer.from_pretrained(
    #     args.model_name_or_path,
    #     do_lower_case=args.do_lower_case,
    #     cache_dir=args.cache_dir if args.cache_dir else None,        
    # )
    tokenizer = LlamaTokenizer.from_pretrained('/home/zikaixiao/zikai/flfm/shepherd/alpaca_native')
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # train_dataset = PromptDataset(args, args.task_name, tokenizer, data_type='train')
    train_dataset = PromptDataset(args, args.task_name, tokenizer, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=8, collate_fn=train_dataset.collate_fn)    
    eval_dataset = PromptDataset(args, args.task_name, tokenizer, data_type='dev')
    train_dataloader_list, eval_dataloader_list, n_sample_list = partition(args, train_dataset, eval_dataset)


    finish = 0