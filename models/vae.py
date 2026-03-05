import torch , math
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim, inchannel=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, 32, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32,  kernel_size=3, padding=1),                 nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),            nn.SiLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),            nn.SiLU()
        )
        self.fc_mu     = nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)

        self.up1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.SiLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.SiLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.SiLU()
        )
        self.final = nn.Conv2d(32, inchannel, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)
        nn.init.zeros_(self.fc_logvar.bias)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        d = self.dec1_conv(self.up1(z))
        d = self.dec2_conv(self.up2(d))
        d = self.dec3_conv(self.up3(d))
        out = self.final(d)
        out = F.interpolate(out, size=(28, 28), mode='bilinear', align_corners=False)
        return torch.clamp(out, -15, 15)

    def forward(self, x):
        e      = self.conv3(self.conv2(self.conv1(x)))
        mu     = self.fc_mu(e)
        logvar = torch.clamp(self.fc_logvar(e), -4, 2)
        z      = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_logits, x, mu, logvar, kl_weight):
    x_01  = (x + 1.0) * 0.5
    recon = F.binary_cross_entropy_with_logits(
                recon_logits, x_01, reduction='sum') / x.shape[0]
    
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon + kl_weight * kl, recon, kl




## 
class LatentVAE(nn.Module):
    def __init__(self , latentdim , inchannel):
        super().__init__()
        self.latent_dim = latentdim
        self.inchannel = inchannel
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(inchannel , 32 , kernel_size = 3 , padding = 1) , nn.SiLU(),
            nn.Conv2d(32 , 32 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )

        self.pool1 = nn.Conv2d(32 , 32 , kernel_size = 3 , stride = 2 , padding = 1)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32 , 64 , kernel_size = 3 , padding = 1) , nn.SiLU() ,
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )

        self.pool2 = nn.Conv2d(64 , 64 , kernel_size = 3 , stride = 2 , padding = 1)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64 , 128 , kernel_size = 3 , padding = 1) , nn.SiLU() ,
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )

        self.pool3 = nn.Conv2d(128 , 128 , kernel_size = 3 , stride = 2 , padding = 1)

        self.mu = nn.Conv2d(128 , self.latent_dim , kernel_size = 1)
        self.var = nn.Conv2d(128 , self.latent_dim , kernel_size = 1)


        # decoder
        self.up1 = nn.ConvTranspose2d(self.latent_dim , 128 , kernel_size = 1 , stride = 2)
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1) , nn.SiLU() ,
            nn.Conv2d(128 , 128 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(128 , 64 , kernel_size = 1 , stride = 2)

        self.dec2 = nn.Sequential(
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1) , nn.SiLU() ,
            nn.Conv2d(64 , 64 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )       

        self.up3 = nn.ConvTranspose2d(64 , 32 , kernel_size = 1 , stride = 2)

        self.dec3 = nn.Sequential(
            nn.Conv2d(32 , 32 , kernel_size = 3 , padding = 1) , nn.SiLU() ,
            nn.Conv2d(32 , 32 , kernel_size = 3 , padding = 1) , nn.SiLU()
        )       

        self.final = nn.Conv2d(32 , inchannel , kernel_size = 1)
    
    def reparametrize(self , mu , logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
    
    def decode(self , x):
        d = self.dec1(self.up1(x))
        d = self.dec2(self.up2(d))
        d = self.dec3(self.up3(d))

        out = self.final(d)
        out = F.interpolate(out , size = (28 , 28) , mode = 'bilinear' , align_corners = False)

        return torch.clamp(out , -15 , 15)
    
    
    def forward(self , x):
        d = self.pool1(self.enc1(x))
        d = self.pool2(self.enc2(d))
        d = self.pool3(self.enc3(d))


        mu = self.mu(d)
        logvar = torch.clamp(self.var(d) , -4 , 2)

        z = self.reparametrize(mu , logvar)
        return self.decode(z) , mu , logvar
