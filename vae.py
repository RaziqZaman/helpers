import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# === VAE Definition ===

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
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        sigma = self.fc_sigma(h)
        return mu, sigma

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
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

# === Loss Function ===

def vae_loss(data, reconstruction, mu, log_var):
    recon_loss = F.binary_cross_entropy(reconstruction, data, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

# === Training Loop ===

def train_vae(model, optimizer, dataloader, n_epochs=50):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            x_hat, mu, sigma = model(x)
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

# === Main Pipeline ===

if __name__ == "__main__":
    # === Load and clean ===
    input_csv = "trips_vae_input_max12.csv"
    output_csv = "augmented_output.csv"

    df = clean_csv_data(input_csv)

    # === Re-identify columns after cleaning ===
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # === Preprocessing ===
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), numerical_cols),
        ("cat", ohe, categorical_cols)
    ])

    processed_data = preprocessor.fit_transform(df)
    processed_tensor = torch.tensor(processed_data, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(processed_tensor), batch_size=64, shuffle=True)

    # === Train VAE ===
    input_dim = processed_tensor.shape[1]
    vae = VAE(input_dim=input_dim, latent_dim=36)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    train_vae(vae, optimizer, dataloader)

    # === Generate synthetic data ===
    n_samples = len(df)
    with torch.no_grad():
        synthetic_data = vae.sample(n_samples).numpy()
        synthetic_data = np.clip(synthetic_data, 0, 1)
        synthetic_original = preprocessor.inverse_transform(synthetic_data)

    # === Combine and save ===
    synthetic_df = pd.DataFrame(synthetic_original, columns=numerical_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)))
    synthetic_df = synthetic_df[df.columns]  # Reorder columns to match original

    # Tag data sources
    df["source"] = "original"
    synthetic_df["source"] = "synthetic"

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    augmented_df.to_csv(output_csv, index=False)
    print(f"✅ Augmented CSV saved to {output_csv}")