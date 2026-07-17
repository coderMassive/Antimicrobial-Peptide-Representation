import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import umap

from group_model import Encoder, Decoder, GroupingAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = sys.argv[1]

npy_folder = "dataset"
output_png = "results.png"
output_umap_png = "umap_grouping.png"


npy_files = sorted(glob.glob(os.path.join(npy_folder, "*.npy")))

X = np.zeros((len(npy_files), 256, 256), dtype=np.float32)

for i, npy_path in enumerate(npy_files):
    X[i] = np.load(npy_path).astype(np.float32)

if X.max() > 1.0:
    X = X / X.max()

_, X_test, _, test_files = train_test_split(
    X,
    npy_files,
    test_size=0.2,
    random_state=42
)

encoder = Encoder()
decoder = Decoder()
model = GroupingAutoencoder(encoder, decoder).to(device)

state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

n = min(10, len(X_test))

inputs = torch.from_numpy(X_test[:n]).unsqueeze(1).to(device)

with torch.no_grad():
    result_imgs = model(inputs).cpu().numpy()

result_imgs = result_imgs.squeeze(1)

plt.figure(figsize=(4 * n, 8))

for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i], cmap="gray")
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(result_imgs[i], cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches="tight")
plt.close()

all_inputs = torch.from_numpy(X_test).unsqueeze(1).to(device)

embeddings = []

batch_size = 16

with torch.no_grad():
    for start in range(0, len(all_inputs), batch_size):
        batch = all_inputs[start:start + batch_size]
        z = model.encode(batch)
        embeddings.append(z.cpu().numpy())

embeddings = np.concatenate(embeddings, axis=0)

print("Embeddings shape:", embeddings.shape)

uniprot_ids_ordered = []
for path in test_files:
    filename = os.path.basename(path)
    base_id = filename.replace(".npy", "")
    header_parts = base_id.split('|')
    uniprot_id = header_parts[1] if len(header_parts) >= 2 else base_id
    uniprot_ids_ordered.append(uniprot_id)

antimicrobial_set = set()
test_ids_set = set(uniprot_ids_ordered)

fasta_path = "uniprot_sprot.fasta"
from Bio import SeqIO

for record in SeqIO.parse(fasta_path, "fasta"):
    parts = record.id.split('|')
    rec_id = parts[1] if len(parts) >= 2 else record.id
    
    if rec_id in test_ids_set:
        if "antimicrobial" in record.description.lower():
            antimicrobial_set.add(rec_id)

test_labels = np.array([1 if uid in antimicrobial_set else 0 for uid in uniprot_ids_ordered])

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
embedding_2d = reducer.fit_transform(embeddings)

plt.figure(figsize=(10, 8))

anti_mask = (test_labels == 1)
not_anti_mask = (test_labels == 0)

plt.scatter(
    embedding_2d[not_anti_mask, 0],
    embedding_2d[not_anti_mask, 1],
    c="#E0E0E0",            
    label="Not Antimicrobial",
    edgecolors='none',
    zorder=1                
)

plt.scatter(
    embedding_2d[anti_mask, 0],
    embedding_2d[anti_mask, 1],
    c="#FF3B30",            
    label=f"Antimicrobial (n={np.sum(anti_mask)})",
    zorder=2                
)

plt.title("UMAP Embeddings Segmented by Antimicrobial Status", fontsize=14, pad=15, fontweight='bold')
plt.xlabel("UMAP Dimension 1", fontsize=11)
plt.ylabel("UMAP Dimension 2", fontsize=11)

plt.legend(loc="best", frameon=True, facecolor="white", edgecolor="#D3D3D3", shadow=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.2)

plt.tight_layout()
plt.savefig(output_umap_png, dpi=200, bbox_inches="tight") # Bumped DPI to 200 for sharper images
plt.close()
