import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from conv import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

npy_folder = "dataset"
checkpoint_path = sys.argv[1]
output_png = "results.png"

npy_files = sorted(glob.glob(os.path.join(npy_folder, "*.npy")))

X = np.zeros((len(npy_files), 256, 256), dtype=np.float32)

for i, npy_path in enumerate(npy_files):
    X[i] = np.load(npy_path).astype(np.float32)

_, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

model = Autoencoder().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

n = 10

inputs = torch.from_numpy(X_test[:n]).unsqueeze(1).to(device)

with torch.no_grad():
    result_imgs = model(inputs).cpu().numpy()

result_imgs = result_imgs.squeeze(1)

plt.figure(figsize=(40, 8))

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.axis("off")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(result_imgs[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches="tight")
plt.close()
