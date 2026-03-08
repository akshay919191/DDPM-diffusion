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


def vae_loss(recon, x, mu, logvar, epoch):
    mse  = F.mse_loss(recon, x)
    ssim_loss = 1 - ssim(recon, x, data_range=1.0, size_average=True)
    recon_loss = mse + 0.5 * ssim_loss
    
    kl_weight = min(epoch / 50, 1.0) * 0.00001
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_weight * kl, recon_loss, kl




## 
class LatentVAE(nn.Module):
    def __init__(self, latentdim, inchannel):
        super().__init__()
        
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.SiLU()
            )
        
        # ── Encoder ──────────────────────────────────────────
        self.enc1 = block(inchannel, 128)
        self.res1 = nn.Sequential(RESIDUAL(128, 128), RESIDUAL(128, 128))

        self.pool1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            RESIDUAL(128, 128),
            nn.SiLU()
        )

        self.enc2 = block(128, 256)
        self.res2 = nn.Sequential(RESIDUAL(256, 256), RESIDUAL(256, 256))

        self.pool2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            RESIDUAL(256, 256),
            nn.SiLU()
        )

        self.enc3 = block(256, 512)
        self.res3 = nn.Sequential(
            RESIDUAL(512, 512),
            RESIDUAL(512, 512),
            RESIDUAL(512, 512)   # extra depth at bottleneck
        )

        self.mu     = nn.Conv2d(512, latentdim, 1)
        self.logvar = nn.Conv2d(512, latentdim, 1)

        # ── Decoder ──────────────────────────────────────────
        self.res_latent = nn.Sequential(
            RESIDUAL(latentdim, latentdim),
            RESIDUAL(latentdim, latentdim)  # refine latent before decoding
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latentdim, 512, 2, stride=2),
            RESIDUAL(512, 512),
            nn.SiLU()
        )
        self.dec1 = block(512, 256)
        self.res4 = nn.Sequential(RESIDUAL(256, 256), RESIDUAL(256, 256))

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            RESIDUAL(128, 128),
            nn.SiLU()
        )
        self.dec2 = block(128, 128)
        self.res5 = nn.Sequential(RESIDUAL(128, 128), RESIDUAL(128, 128))

        self.final = nn.Sequential(
            nn.Conv2d(128, inchannel, 1),
            nn.Tanh() 
        )

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.res1(self.enc1(x))          # [B, 128, 32, 32]
        x = self.res2(self.enc2(self.pool1(x)))  # [B, 256, 16, 16]
        x = self.res3(self.enc3(self.pool2(x)))  # [B, 512,  8,  8]
        return x

    def decode(self, z):
        z = self.res_latent(z)               # [B, latentdim, 8, 8]
        z = self.res4(self.dec1(self.up1(z)))    # [B, 256, 16, 16]
        z = self.res5(self.dec2(self.up2(z)))    # [B, 128, 32, 32]
        return self.final(z)                

    def forward(self, x):
        b      = self.encode(x)
        mu     = self.mu(b)
        logvar = torch.clamp(self.logvar(b), -10, 10)
        z      = self.reparametrize(mu, logvar)
        recon  = self.decode(z)
        return recon, mu, logvar
