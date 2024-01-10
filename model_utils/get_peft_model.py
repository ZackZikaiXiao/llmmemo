from peft import (
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    IA3Config,
    set_peft_model_state_dict,
)
import torch
class PeftHelper():
    def __init__(self,model_name, peft_method):
        self.model_name = model_name
        self.peft_method = peft_method

    def get_peft_model_for_training(self, args ,model):
        if self.model_name == 'alpaca' or self.model_name == 'Llama2-7B':
            if self.peft_method == 'lora':
                return get_lora_peft_model(args, model)
            elif self.peft_method == 'prefix_tuning':
                return get_prefix_tuning_peft_model(args, model)
            elif self.peft_method == 'IA3':
                return get_IA3_peft_model(args, model)
            
    def get_peft_model_for_inference(self, model, config_path, weight_path):
        if self.peft_method == 'lora':
            config = LoraConfig.from_pretrained(config_path)
        elif self.peft_method == 'prefix_tuning':
            config = PrefixTuningConfig.from_pretrained(config_path)
        config.inference_mode = True
        peft_weights = torch.load(weight_path)
        model = get_peft_model(model, config)
        set_peft_model_state_dict(model, peft_weights, "default")
        del peft_weights
        return model

        


def get_lora_peft_model(args, model):
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    return model, config

def get_prefix_tuning_peft_model(args, model):
    config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        # inference_mode=False,
        num_virtual_tokens=args.num_virtual_tokens,
    )
    model = get_peft_model(model, config)
    return model, config

def get_IA3_peft_model(args, model):
    config = IA3Config(
        task_type="CAUSAL_LM", 
        target_modules=["k_proj", "v_proj", "down_proj"], 
        feedforward_modules=["down_proj"],
        )
    model = get_peft_model(model, config)
    return model, config

if __name__ == "__main__":
    config = LoraConfig(
        bias="none",
        task_type="CAUSAL_LM",
    )
    print('a')
    