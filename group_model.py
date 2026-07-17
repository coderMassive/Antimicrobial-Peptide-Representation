import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)

        # 256 -> 128 -> 64 -> 32
        self.bottleneck = nn.Linear(32 * 32 * 128, 1024)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = F.relu(self.bottleneck(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(1024, 32 * 32 * 128)

        self.decoder = nn.Sequential(
            nn.Dropout(0.2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 128, 32, 32)
        x = self.decoder(x)
        return x

class GroupingAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, grouping_weight=0.5, margin=1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.grouping_weight = grouping_weight
        self.margin = margin
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def compute_losses(self, x):
        noise = torch.randn_like(x) * 0.05
        x_pos = torch.clamp(x + noise, 0.0, 1.0)

        x_neg = torch.roll(x, shifts=1, dims=0)

        z = self.encoder(x)
        z_pos = self.encoder(x_pos)
        z_neg = self.encoder(x_neg)

        reconstructed = self.decoder(z)

        recon_loss = self.loss_fn(reconstructed, x)

        d_pos = torch.sum((z - z_pos) ** 2, dim=1)
        d_neg = torch.sum((z - z_neg) ** 2, dim=1)

        grouping_loss = torch.mean(
            torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        )

        total_loss = recon_loss + self.grouping_weight * grouping_loss

        return total_loss, recon_loss, grouping_loss

npy_folder = "dataset"
npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

X = np.zeros((len(npy_files), 256, 256), dtype=np.float32)

for i, npy_path in enumerate(npy_files):
    X[i] = np.load(npy_path).astype(np.float32)

X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)

print("X_train_torch:", X_train_torch.shape)
print("X_test_torch:", X_test_torch.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = TensorDataset(X_train_torch)
val_dataset = TensorDataset(X_test_torch)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)

encoder = Encoder()
decoder = Decoder()

grouping_model = GroupingAutoencoder(
    encoder,
    decoder,
    grouping_weight=1.0,
    margin=2.0
).to(device)

optimizer = torch.optim.Adam(grouping_model.parameters())

epochs = 100
patience = 10

best_val_loss = float("inf")
best_state = None
patience_counter = 0

history_grouping = {
    "loss": [],
    "recon_loss": [],
    "group_loss": [],
    "val_recon_loss": []
}

for epoch in range(epochs):
    grouping_model.train()

    train_total_loss = 0.0
    train_recon_loss = 0.0
    train_group_loss = 0.0

    for batch in train_loader:
        x = batch[0].to(device)

        optimizer.zero_grad()

        total_loss, recon_loss, group_loss = grouping_model.compute_losses(x)

        total_loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        train_total_loss += total_loss.item() * batch_size
        train_recon_loss += recon_loss.item() * batch_size
        train_group_loss += group_loss.item() * batch_size

    train_total_loss /= len(train_loader.dataset)
    train_recon_loss /= len(train_loader.dataset)
    train_group_loss /= len(train_loader.dataset)

    grouping_model.eval()
    val_recon_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)

            reconstructed = grouping_model(x)
            recon_loss = grouping_model.loss_fn(reconstructed, x)

            val_recon_loss += recon_loss.item() * x.size(0)

    val_recon_loss /= len(val_loader.dataset)

    history_grouping["loss"].append(train_total_loss)
    history_grouping["recon_loss"].append(train_recon_loss)
    history_grouping["group_loss"].append(train_group_loss)
    history_grouping["val_recon_loss"].append(val_recon_loss)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- loss: {train_total_loss:.4f} "
        f"- recon_loss: {train_recon_loss:.4f} "
        f"- group_loss: {train_group_loss:.4f} "
        f"- val_recon_loss: {val_recon_loss:.4f}"
    )

    if val_recon_loss < best_val_loss:
        best_val_loss = val_recon_loss
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in grouping_model.state_dict().items()
        }
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

if best_state is not None:
    grouping_model.load_state_dict(best_state)

torch.save(grouping_model.state_dict(), "grouping_autoencoder.pt")
print("Saved model to grouping_autoencoder.pt")
