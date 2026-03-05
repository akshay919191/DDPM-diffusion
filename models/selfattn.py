import torch , math
import torch.nn as nn

class MULTIHEADATTN(nn.Module):
    def __init__(self , embed_dim , n_head):
        super().__init__()
        self.embed = embed_dim
        self.num_head = n_head

        assert (embed_dim % n_head == 0) , "try different"

        self.d_k = embed_dim // n_head

        self.wq = nn.Parameter(torch.randn(embed_dim , embed_dim , device = 'cuda'))
        self.wk = nn.Parameter(torch.randn(embed_dim , embed_dim , device = 'cuda'))
        self.wv = nn.Parameter(torch.randn(embed_dim , embed_dim , device = 'cuda'))

        self.baisq = nn.Parameter(torch.randn(embed_dim , device = 'cuda'))
        self.baisk = nn.Parameter(torch.randn(embed_dim , device = 'cuda'))
        self.biasv = nn.Parameter(torch.randn(embed_dim , device = 'cuda'))

        self.wo = nn.Parameter(torch.randn((embed_dim , embed_dim), device = 'cuda'))

        for elements in [self.wq , self.wk , self.wv , self.wo]:
            nn.init.kaiming_uniform_(elements)

        for elements in [self.baisq , self.baisk , self.biasv]:
            nn.init.zeros_(elements)

    @staticmethod
    def Attention(q , k , v):
        d_k = q.shape[-1]

        score = (q@k.transpose(-1 , -2)) / math.sqrt(d_k)
        score = score.softmax(dim = -1)

        attnscore = score @ v

        return attnscore

    def forward(self , x): # i will use only one input as its obvious that its self attn

        query = x @ self.wq + self.baisq
        key = x @ self.wk + self.baisk
        value = x @ self.wv + self.biasv

        batch , seq , _ = query.shape

        # shape is batch , seq , embed

        q = query.view(batch , seq , self.num_head , self.d_k).permute(0 , 2 , 1 , 3)
        k = key.view(batch , seq , self.num_head , self.d_k).permute(0 , 2 , 1 , 3)
        v = value.view(batch , seq , self.num_head , self.d_k).permute(0 , 2 , 1 , 3)

        attnscore = self.Attention(q , k , v)

        out = attnscore.contiguous().view(batch , seq , -1)

        return out @ self.wo
    
class AttentionWrapper(nn.Module):
    def __init__(self, channels, n_head):
        super().__init__()
        self.channels = channels
        self.attn = MULTIHEADATTN(embed_dim=channels, n_head=n_head)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1)
        
        x_norm = self.norm(x_flat)
        out = self.attn(x_norm)
        
        out = out + x_flat 
        
        return out.permute(0, 2, 1).view(b, c, h, w)