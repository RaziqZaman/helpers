import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# === VAE Class Definition ===

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
        )
        
        # Mean and log-variance outputs
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_sigma = nn.Linear(64, latent_dim)
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Activation for reconstruction
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)  # Variance = exp(sigma)
        eps = torch.randn_like(std)   # Random noise
        return mu + eps * std  # Latent variable z

    def decode(self, z):
        return self.decoder(z)  # Reconstructed input

    def forward(self, x):
        # Encode to get mu and sigma
        mu, sigma = self.encode(x)
        
        # Sample from the latent space using the reparameterization trick
        z = self.reparameterize(mu, sigma)
        
        # Decode the sampled z to get the reconstruction
        x_hat = self.decode(z)
        
        # Return reconstruction, mu, and sigma
        return x_hat, mu, sigma

    @torch.no_grad()
    def sample(self, n):
        self.eval()
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)
# === Loss Function ===

def vae_loss(data, reconstruction, mu, log_var):
    recon_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

# === Training Loop ===

def train_vae(model, optimizer, dataloader, n_epochs=120):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            x = x.view(x.size(0), -1)  # Flatten the data
            optimizer.zero_grad()
            x_hat, mu, sigma = model(x)  # Unpack the 3 outputs
            loss = vae_loss(x, x_hat, mu, sigma)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {total_loss:.2f}")

# === Cleaning Utility ===

def clean_csv_data(file_path, fill_missing_numeric=0, fill_missing_categorical="unknown"):
    print(f"📂 Loading CSV: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"✅ Converted {col} to numeric")
            except ValueError:
                print(f"🔡 Keeping {col} as categorical")

    # Fill missing values
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(fill_missing_numeric)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df[cat_cols] = df[cat_cols].fillna(fill_missing_categorical)

    print("🧽 Cleaning complete.")
    return df

# === Parameters for Customization ===

# Set values for the parameters directly in the script
latent_dim = 108  # Size of the latent dimension
n_samples = 110880  # Number of synthetic samples to generate (population of Maryland in 2018 according to MARYLAND VITAL STATISTICS ANNUAL REPORT)
input_csv = "trips_vae_input_max12.csv"  # Path to your input CSV file
output_csv = "trips_vae_output_raw_110880.csv"  # Path to the output augmented CSV file

# === Load and clean ===
df = clean_csv_data(input_csv)

# === Re-identify columns after cleaning ===
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

# === Preprocessing ===
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numerical_cols),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
])

processed_data = preprocessor.fit_transform(df)
processed_tensor = torch.tensor(processed_data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(processed_tensor), batch_size=64, shuffle=True)

# === Train VAE ===
input_dim = processed_tensor.shape[1]
vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
train_vae(vae, optimizer, dataloader)

# === Generate synthetic data ===
with torch.no_grad():
    synthetic_data = vae.sample(n_samples).numpy()
    synthetic_data = np.clip(synthetic_data, 0, 1)

# === Manually Inverse Transform the Data ===

# Get transformer objects
ohe = preprocessor.named_transformers_["cat"]
scaler = preprocessor.named_transformers_["num"]

# Split synthetic data back into numeric and categorical parts
num_features = len(numerical_cols)
synthetic_num = synthetic_data[:, :num_features]
synthetic_cat = synthetic_data[:, num_features:]

# Inverse transform the numeric and categorical parts separately
decoded_num = scaler.inverse_transform(synthetic_num)
decoded_cat = ohe.inverse_transform(synthetic_cat)

# Recombine numeric and categorical data
decoded_full = np.concatenate([decoded_num, decoded_cat], axis=1)

# Create DataFrame for the decoded synthetic data
decoded_df = pd.DataFrame(decoded_full, columns=numerical_cols + categorical_cols)
decoded_df["source"] = "synthetic"

# === Combine and save ===
df["source"] = "original"
augmented_df = pd.concat([df, decoded_df], ignore_index=True)
augmented_df.to_csv(output_csv, index=False)
print(f"✅ Augmented CSV saved to {output_csv}")
