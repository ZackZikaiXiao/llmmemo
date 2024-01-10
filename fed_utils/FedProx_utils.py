import torch
from transformers import Trainer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from collections import OrderedDict
def compute_proximal_term(original_peft_weights, updated_model, device):
    # define how to compute difference loss
    proximal_term = 0.0
    new_peft_weights = OrderedDict((name, param) for name, param in updated_model.named_parameters() if "default" in name)
    for original_weights, updated_weights in zip(original_peft_weights.values(), new_peft_weights.values()):
        original_weights = original_weights.to(device)
        updated_weights = updated_weights.to(device)
        proximal_term += torch.norm((original_weights - updated_weights), p=2)
    return proximal_term

    
class FedProxTrainer(Trainer):
    def set_previous_peft_weights(self, peft_weights):
        self.previous_peft_weights = peft_weights

    def set_proximal_term_mu(self, arg):
        self.proximal_term_mu = arg
        
    def compute_loss(self, model, inputs, return_outputs=False):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)
        loss_device = loss.device
        proximal_term = compute_proximal_term(original_peft_weights=self.previous_peft_weights, updated_model=model, device=loss_device)
        proximal_term = proximal_term.to(loss_device)
        # print("proximal term loss: " + str(0.5 * self.proximal_term_mu * proximal_term))
        loss += 0.5 * self.proximal_term_mu * proximal_term
        return (loss, outputs) if return_outputs else loss