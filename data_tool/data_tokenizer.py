from utils.prompter import Prompter

class DataTokenizer:
    def __init__(self, args, tokenizer) -> None:
        self.args = args
        self.dataset = args.dataset
        self.tokenizer = tokenizer
        self.prompter = Prompter(args.prompt_template_name)
        
    def generate_and_tokenize_prompt(self, data_point):
        GLUE_dataset = ['sst-2', 'rte', 'cola', 'wnli', 'sts-b', 'mnli', 'mrpc', 'qnli', 'qqp']
        if self.dataset == "new-databricks-dolly-15k":
            return self._generate_and_tokenize_prompt_new_databricks_dolly_15k(data_point)
        elif self.dataset in GLUE_dataset:
            return self._generate_and_tokenize_GLUE(data_point)
        elif self.dataset == "quail":
            return self._generate_and_tokenize_GLUE(data_point)
    
    # 两步:1.把形式化json转化成完整段落的prompt； 2.将prompt进行tokenize
    def _generate_and_tokenize_prompt_new_databricks_dolly_15k(self, data_point):
        # 结合template，把json转化为paragraph
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )

        # tokenization
        tokenized_full_prompt = self._tokenize_new_databricks_dolly_15k(full_prompt)   # {input_ids; attention_mask; labels}
        if not self.args.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = self._tokenize_new_databricks_dolly_15k(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt
    
    
    def _tokenize_new_databricks_dolly_15k(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            text=prompt,
            # text_target=target,
            truncation=True,
            max_length=self.args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                # make sure the last id is eos id, otherwise, model doesn't know when to stop
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
    def _generate_and_tokenize_GLUE(self, data_point):
        # 结合template，把json转化为paragraph
        
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["context"],
            data_point["response"],
        )
        # target = data_point["response"]

        # tokenization
        tokenized_full_prompt = self._tokenize_GLUE(full_prompt)   # {input_ids; attention_mask; labels}
        if not self.args.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["context"]
            )
            tokenized_user_prompt = self._tokenize_GLUE(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # 为什么是-100填充
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
        return tokenized_full_prompt
    
    
    def _tokenize_GLUE(self, prompt, add_eos_token=True):
        result = self.tokenizer(
            text=prompt,
            # text_target=target,
            truncation=True,
            max_length=self.args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result
    
