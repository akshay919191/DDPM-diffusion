import torch , math
import torch.nn as nn

device = 'cuda'

def scheduler(timesteps = 1000):
    beta = torch.linspace(1e-4 , 0.02 , timesteps , device = device)

    alpha = 1.0 - beta
    alpha_hat = torch.cumprod(alpha , dim = 0)

    alpha_hat_sqrt = torch.sqrt(alpha_hat)
    one_minus_hat_sqrt = torch.sqrt(1.0 - alpha_hat)

    return {
        'beta' : beta ,
        'alpha' : alpha ,
        'alpha_hat' : alpha_hat ,
        'alpha_hat_sqrt' : alpha_hat_sqrt ,
        'one_minus_hat_sqrt' : one_minus_hat_sqrt
    }


sched = scheduler()
one_minus_hat_sqrt = sched['one_minus_hat_sqrt']
alpha_hat_sqrt = sched['alpha_hat_sqrt']
alpha = sched['alpha']
alpha_hat = sched['alpha_hat']


def ddim_step(x_t, pred_noise, t, eta=0.0):
    t_prev = max(t - 1, 0)

    alpha_t = alpha_hat[t]
    alpha_prev = alpha_hat[t_prev]

    x0_pred = (x_t - one_minus_hat_sqrt[t] * pred_noise) / alpha_hat_sqrt[t]

    sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
    dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise

    noise = torch.zeros_like(x_t) if eta == 0 else torch.randn_like(x_t)
    x_prev = alpha_hat_sqrt[t_prev] * x0_pred + dir_xt + sigma * noise

    return x_prev
