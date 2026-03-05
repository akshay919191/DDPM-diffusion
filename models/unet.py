import torch , math
import torch.nn as nn
import torch.nn.functional as F
from models.selfattn import AttentionWrapper
## 
class RESIDUAL(nn.Module):
    def __init__(self , inchannel , outchannel , timeinjectdim):
        super().__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel
        self.timeinjectidm = timeinjectdim

        # block 1
        self.grp1 = nn.GroupNorm(num_groups = 8 , num_channels = inchannel)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(inchannel , outchannel , kernel_size = 3 , padding = 1) # batch , channel , in , out

        self.timeproj = nn.Linear(timeinjectdim , outchannel) # time , out

        # block 2
        self.grp2 = nn.GroupNorm(num_groups = 8 , num_channels = outchannel)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(outchannel , outchannel , kernel_size = 3 , padding = 1) # batch , channel , in , out

        # shortcut connection for fallback

        if inchannel != outchannel:
            self.shortcut = nn.Conv2d(inchannel , outchannel , kernel_size = 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self , x , t_emb):
        h = self.conv1(self.act1(self.grp1(x))) # batch , channel , in , out

        time = self.timeproj(self.act1(t_emb)) # timeinject , out
        h = h + time[: , : , None , None] # batch , channel , in , out

        h = self.conv2(self.act2(self.grp2(h))) # batch , channel , in , out

        return h + self.shortcut(x)

    

class UNET(nn.Module):
    def __init__(self, in_channels , timedim):
        super().__init__()
        self.timedim = timedim

        self.label_emb = nn.Embedding(11 , embedding_dim = timedim)

        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.timeMLP1 = nn.Sequential(
            nn.Linear(timedim , 64),
            nn.SiLU() ,
            nn.Linear(64 , 64)
        )

        self.pool1 = nn.Conv2d(64, 64, kernel_size=2, stride=2) # 28 -> 14


        self.enc2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU()
        )
        self.timeMLP2 = nn.Sequential(
            nn.Linear(timedim , 128),
            nn.SiLU() ,
            nn.Linear(128 , 128)
        )

        self.pool2 = nn.Conv2d(128, 128, kernel_size=2, stride=2) # 14 -> 7


        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            AttentionWrapper(256 , 8),
        )

        self.timeMLP3 = nn.Sequential(
            nn.Linear(timedim , 256),
            nn.SiLU() ,
            nn.Linear(256 , 256)
        )

        self.up1 = nn.ConvTranspose2d(256 , 128 , kernel_size = 2 , stride = 2)

        self.up1_conv = nn.Sequential(
            nn.Conv2d(256 , 128 , kernel_size = 3 , padding = 1),
            nn.SiLU(),
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1),
            nn.SiLU()
        )
        self.timeMLP4 = nn.Sequential(
            nn.Linear(timedim , 128),
            nn.SiLU() ,
            nn.Linear(128 , 128)
        )


        self.up2 = nn.ConvTranspose2d(128 , 64 , kernel_size = 2 , stride = 2)

        self.up2_conv = nn.Sequential(
            nn.Conv2d(128 , 64 , kernel_size = 3 , padding = 1),
            nn.SiLU() ,
            nn.Conv2d(64 ,  64 , kernel_size = 3 , padding = 1),
            nn.SiLU()
        )
        self.timeMLP5 = nn.Sequential(
            nn.Linear(timedim , 64),
            nn.SiLU() ,
            nn.Linear(64 , 64)
        )


        self.final = nn.Conv2d(64 , in_channels , kernel_size = 1 , padding = 0)

    def forward(self , x , t , y):
        t_emb = get_time_embedding(t, self.timedim, device=t.device)
        label_emb = self.label_emb(y)
        time_embed = t_emb + label_emb
        
        time1_4d = self.timeMLP1(time_embed)[:, :, None, None]
        time2_4d = self.timeMLP2(time_embed)[:, :, None, None]
        time3_4d = self.timeMLP3(time_embed)[:, :, None, None]
        time4_4d = self.timeMLP4(time_embed)[:, :, None, None]
        time5_4d = self.timeMLP5(time_embed)[:, :, None, None]



        enc1 = self.enc1_conv(x) + time1_4d
        
        #pool1 = self.pool1(enc1)

        enc2 = self.enc2_conv(enc1) + time2_4d
        #pool2 = self.pool2(enc2)

        bottleneck = self.bottleneck(enc2) + time3_4d

        dec1 = self.up1(bottleneck) + time4_4d
        dec1 = F.interpolate(dec1, size=enc2.shape[2:], mode='nearest')
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.up1_conv(dec1)

        dec2 = self.up2(dec1) + time5_4d
        dec2 = F.interpolate(dec2, size=enc1.shape[2:], mode='nearest')
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.up2_conv(dec2)

        final = self.final(dec2) 

        return final
    
    

def get_time_embedding(timesteps, embedding_dim, device=None):

    if device is None:
        device = timesteps.device  

    half_dim = embedding_dim // 2

    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
    
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

    return emb  

