import math
def cosine_annealing_LR(T_max, epoch, initial_lr, eta_min=0):
    new_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(epoch * math.pi / T_max))
    return new_lr

def cosine_annealing_warm_restart_LR(T_max, epoch, initial_lr, eta_min=0):
    epoch = epoch % T_max
    new_lr = eta_min + 0.5 * (initial_lr - eta_min) * (1 + math.cos(epoch * math.pi / T_max))
    return new_lr