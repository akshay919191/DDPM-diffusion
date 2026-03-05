
import torch , math
import torch.nn as nn 
from models.scheduler import scheduler , step
from models.unet import UNET
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from models.vae import LatentVAE
from models.unet import UNET
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

device = 'cuda'
sched = scheduler()
model = UNET()

device = 'cuda'
model = LatentVAE(latentdim = 8 , inchannel = 1).to(device)
optimizer = torch.optim.Adam(model.parameters() , lr = 1e-4)
beta_final = 1.0
anneal_step = 28140
global_step = 0

def vaeLOSS(recon_logits , x , mu , logvar , kl_weight):
    x0 = (x + 1) * 0.5

    recon = F.binary_cross_entropy_with_logits(recon_logits , x0 , reduction = 'sum') / x.shape[0]

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]

    return recon + kl * kl_weight , recon , kl


from tqdm import tqdm
@torch.no_grad()
def sample_digits(model, vae, scheduler, n=10, device='cuda'):
    model.eval()
    vae.eval()

    n = 10
    x = torch.randn(n, 8, 4, 4, device=device)

    mixed_labels = torch.tensor([3, 8] * (n // 2) + [3]*(n % 2), device=device).long()  


    for i in tqdm(reversed(range(1000)), total=1000):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        predicted_noise = model(x, t, y=mixed_labels)  
        x = scheduler(x, predicted_noise, t)

    images = vae.decode(x)
    return images


device = 'cuda' 
unet = UNET(8 , 256).to(device)
optimizer = torch.optim.Adam(unet.parameters() , lr = 1e-4)

loss_running = 0.0

image = sample_digits(unet , model , step)
fig, axis = plt.subplots(1, 10, figsize=(15, 2))  
for i in range(10):
    axis[i].imshow(image[i].cpu().squeeze(), cmap='gray')  
    axis[i].axis('off')  
plt.show()
