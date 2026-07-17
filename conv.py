import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NPYAutoencoderDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays.astype(np.float32)

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        x = self.arrays[idx]

        x = torch.from_numpy(x).unsqueeze(0)

        return x, x

class Autoencoder(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.MaxPool2d(2),  # 32x32
        )

        self.flatten = nn.Flatten()

        self.bottleneck = nn.Sequential(
            nn.Linear(32 * 32 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.decoder_dense = nn.Sequential(
            nn.Linear(1024, 32 * 32 * 128),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

        self.decoder_conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 64x64

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 128x128

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_p),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 256x256

            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.bottleneck(x)

        x = self.decoder_dense(x)
        x = x.view(-1, 128, 32, 32)

        x = self.decoder_conv(x)
        return x

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

train_dataset = NPYAutoencoderDataset(X_train)
test_dataset = NPYAutoencoderDataset(X_test)

batch_size = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

model = Autoencoder().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(test_loader.dataset)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- train_loss: {train_loss:.6f} "
        f"- val_loss: {val_loss:.6f}"
    )

torch.save(model.state_dict(), "autoencoder_pytorch.pt")
