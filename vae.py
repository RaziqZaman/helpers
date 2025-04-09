import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Architecture from https://arxiv.org/abs/2208.01403

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
        n_epochs: int = 120,
        save_epochs: int = 12,
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

### processing

# === 1. Load your CSV ===
df = pd.read_csv("trips_vae_input_max12.csv")

# === 2. Identify categorical and numerical columns ===
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

# === 3. Create transformer ===
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numerical_cols),
    ("cat", OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_cols)
])

# === 4. Fit and transform data ===
processed_data = preprocessor.fit_transform(df)
processed_tensor = torch.tensor(processed_data, dtype=torch.float32)

# === 5. Prepare DataLoader ===
dataset = TensorDataset(processed_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 6. Define and train the VAE ===
from vae_model import VAE, train_vae  # Assuming you saved your model in vae_model.py
input_dim = processed_tensor.shape[1]
vae = VAE(input_dim=input_dim, latent_dim=32)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
train_vae(vae, optimizer, dataloader, n_epochs=50)

# === 7. Sample synthetic data ===
vae.eval()
with torch.no_grad():
    n_synthetic = len(df)
    synthetic = vae.sample(n_synthetic).numpy()
    synthetic = np.clip(synthetic, 0, 1)  # just in case some values go out of range

# === 8. Inverse transform to original space ===
synthetic_original = preprocessor.inverse_transform(synthetic)

# === 9. Create and save final augmented CSV ===
synthetic_df = pd.DataFrame(synthetic_original, columns=numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)))
augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
augmented_df.to_csv("augmented_output.csv", index=False)