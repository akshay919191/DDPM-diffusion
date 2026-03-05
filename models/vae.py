import torch , math
import torch.nn as nn
import torch.nn.functional as F



class VAE(nn.Module):
    def __init__(self, latent_dim, inchannel=1):
        super().__init__()
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU()
        )
        self.fc_mu = nn.Conv2d(128, latent_dim, 3, padding=1)
        self.fc_logvar = nn.Conv2d(128, latent_dim, 3, padding=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU()
        )
        self.final = nn.Conv2d(32, inchannel, 1)

    def encode(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        mu = self.fc_mu(e3)
        logvar = self.fc_logvar(e3)

        logvar = torch.clamp(logvar, -10, 10)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + std * eps

    def decode(self, z):
        d1 = self.up1(z)
        d1 = self.dec1_conv(d1)
        d2 = self.up2(d1)
        d2 = self.dec2_conv(d2)
        d3 = self.up3(d2)
        d3 = self.dec3_conv(d3)
        out = self.final(d3)
        out = F.interpolate(out, size=(28,28), mode='bilinear', align_corners=False)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar

# i started this gangsta shit and here's the mf thanks i get 
