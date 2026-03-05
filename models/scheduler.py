
import torch , math
import torch.nn as nn
import torch.nn.functional as F

##
# scheduler
def scheduler(timesteps = 1000):
    beta = torch.linspace(1e-4 , 0.02 , timesteps , device = 'cuda')
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha , dim = 0)

    alpha_hat_sqrt = torch.sqrt(alpha_hat)
    OneMinusHatSqrt = torch.sqrt(1 - alpha_hat)

    return {
        'beta' : beta,
        'alpha' : alpha ,
        'alpha_hat' : alpha_hat ,
        'alphahataqrt' : alpha_hat_sqrt ,
        'oneminussqrthat' : OneMinusHatSqrt
    }



## 

sched = scheduler()

def nosieADD(x0, noise, t):

    device = x0.device   

    s1 = sched['alphahataqrt'].to(device)[t].view(-1,1,1,1)
    s2 = sched['oneminussqrthat'].to(device)[t].view(-1,1,1,1)

    xt = s1 * x0 + s2 * noise

    return xt, noise


def step(x_t, predicted_noise, t):

    beta_t = sched['beta'][t].view(-1, 1, 1, 1)
    alpha_t = sched['alpha'][t].view(-1, 1, 1, 1)
    alpha_bar_t = sched['alpha_hat'][t].view(-1, 1, 1, 1)
    
    coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
    mean = (1 / torch.sqrt(alpha_t)) * (x_t - coeff * predicted_noise)
    
    if t[0] > 0:
        noise = torch.randn_like(x_t)
        
        sigma_t = torch.sqrt(beta_t) 
        return mean + sigma_t * noise
    else:
        return mean 
