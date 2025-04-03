import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Architecture from https://arxiv.org/abs/2208.01403
# testing

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_sigma = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Assuming the output requires softmax
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)  # sigma is log(variance)
        eps = torch.randn_like(std) 
        return mu + eps * std 

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z), mu, sigma
    
    @torch.no_grad()
    def sample(self, n):
        self.eval()
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


def vae_loss(
        data: torch.Tensor,
        reconstruction: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor
    ):
    recon_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence


def train_vae(
        model: VAE, 
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader, 
        n_epochs: int = 100,
        save_epochs: int = 10,
        save_path: str = "vae.pt"
    ):
    model.train()
    losses = []
    for epoch in range(n_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            x = data
            x = x.view(x.size(0), -1)  # Flatten the data
            x_hat, mu, sigma = model(x)
            loss = vae_loss(x, x_hat, mu, sigma)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if epoch % save_epochs == 0:
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch} |loss: {loss.item()}")
    