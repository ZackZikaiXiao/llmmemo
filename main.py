import os
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

# offline
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

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
    model = prepare_model_for_kbit_training(model)
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
    start_epoch = 0
    if(args.resume_from_checkpoint):
        parameter_path = './lora-shepherd-7b/quail-iid-10/4/adapter_model.bin'
        peft_weights = torch.load(parameter_path)
        set_peft_model_state_dict(model, peft_weights,"default")
        start_epoch = 5
        

    print("The process of federated instruction-tuning has started..")
    previously_selected_clients_set = set()
    last_client_id = None
    local_dataset_len_dict = dict()
    output_dir = args.output_dir
    T_max = args.num_communication_rounds // 4
    training_start_time = time.time()
    for epoch in tqdm(range(start_epoch, args.num_communication_rounds)):

        print("\nConducting the client selection")
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac, args.client_selection_strategy,
                                                other_info=epoch)
        local_learning_rate = args.local_learning_rate
        # local_learning_rate = cosine_annealing_warm_restart_LR(T_max, epoch, args.local_learning_rate)
        print("learning rate of current communication: " + str(local_learning_rate))
        for client_id in selected_clients_set:
            client = GenerateClient(args, client_id, model, output_dir)

            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.load_raw_load(dataset=args.dataset)
            client.preprare_local_dataset(data_tokenizer.generate_and_tokenize_prompt, args.local_val_set_size)
            client.build_local_trainer(tokenizer,
                                       args.local_micro_batch_size,
                                       gradient_accumulation_steps,
                                       args.local_num_epochs,
                                       local_learning_rate,
                                       args.group_by_length,
                                       ddp)

            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()

            print("Local training starts ... ")
            client.train()

            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set, last_client_id = client.terminate_local_training(
                epoch, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        model = FedAvg(model,
                       selected_clients_set,
                       output_dir,
                       local_dataset_len_dict,
                       epoch,
                       )

        torch.save(get_peft_model_state_dict(model), os.path.join(output_dir, str(epoch), "adapter_model.bin"))
        config.save_pretrained(output_dir)
        print("END OF COMMUNICATION: " + str(epoch))
    training_over_time = time.time()
    training_time = int(round((training_over_time - training_start_time)))
    print("Total training time: " + str(datetime.timedelta(seconds = training_time)))


if __name__ == "__main__":
    args = parse_args()
    # partition_data(args)     
    main(args)
    batch_eva_write_to_excel(args.num_communication_rounds, args)
    