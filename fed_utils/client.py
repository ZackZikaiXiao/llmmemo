import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from data_tool.data_path import data_path
from fed_utils.FedProx_utils import FedProxTrainer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


class GenerateClient:
    def __init__(self, args, client_id, model, output_dir):
        self.args = args
        self.client_id = client_id
        self.model = model
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.data_path = args.data_path
    def load_raw_load(self, dataset):
        self.local_data_path = os.path.join(self.data_path, "local_training_{}.json".format(self.client_id))
        # self.local_data_path = os.path.join(data_path[dataset], str(self.args.num_clients), "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        
    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if 'sts-b' in self.local_data_path:
            cols = ['instruction', 'response', 'context', 'category', 'label']
        else:
            cols = ['instruction', 'response', 'context', 'category']
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
            )
        else:
            # test_data = generate_and_tokenize_prompt(self.local_data["train"][0])
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=200,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        if self.args.useFedProx:
            self.local_trainer = FedProxTrainer(
                model=self.model,
                train_dataset=self.local_train_dataset,
                eval_dataset=self.local_eval_dataset,
                args=self.train_args,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )
        else:
            self.local_trainer = transformers.Trainer(
                model=self.model,
                train_dataset=self.local_train_dataset,
                eval_dataset=self.local_eval_dataset,
                args=self.train_args,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
                ),
            )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(get_peft_model_state_dict(self.model))

    def train(self):
        if self.args.useFedProx:
            self.local_trainer.set_previous_peft_weights(self.params_dict_old)
            self.local_trainer.set_proximal_term_mu(self.args.proximal_term_argument)
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):

        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = get_peft_model_state_dict(self.model)
        # new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        # older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, self.params_dict_old, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
