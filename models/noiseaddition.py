import torch , math
import torch.nn as nn
import torch.nn.functional as F
from scheduler import scheduler
## 

sched = scheduler()

def nosieADD(x0 ,  t):
    noise = torch.randn_like(x0 , device = 'cuda')

    s1 = sched['alphahataqrt'][t].view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
    s2 = sched['oneminussqrthat'][t].view(-1, 1, 1, 1)

    xt = s1 * x0 + s2 * noise

    return xt , noise