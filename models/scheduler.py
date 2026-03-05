import torch , math 
import torch.nn as nn 


##
# scheduler
def scheduler(timesteps = 1000):
    beta = torch.linspace(1e-4 , 0.02 , 1000)
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



