import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models import ContactGANFingerprintNet
from peptide_data import DATASET_FILE, load_sequences


def find_dataset(path):
    candidates = [
        Path(path),
        Path.cwd() / DATASET_FILE,
        Path(__file__).resolve().parent / DATASET_FILE,
        Path(__file__).resolve().parent.parent / DATASET_FILE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f'Could not find {DATASET_FILE}. Pass its path with --dataset.')


def load_amp_labels(dataset_path):
    peptides = load_sequences(dataset_path)
    return {
        peptide['id']: 0 if 'non' in str(peptide['label']).lower() else 1
        for peptide in peptides
    }


def parse_ca_coordinates(pdb_path: str):
    coords = []
    with open(pdb_path, 'rb') as handle:
        for line in handle:
            if not line.startswith(b'ATOM  '):
                continue
            if line[13:15] != b'CA':
                continue
            try:
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue

    if len(coords) < 4:
        return None

    return torch.tensor(coords, dtype=torch.float32)


def build_contact_map(coords: torch.Tensor, map_size: int, cutoff: float):
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    distances = torch.sqrt(torch.clamp((diff ** 2).sum(dim=-1), min=1e-12))
    contact = (distances <= cutoff).float()
    contact.fill_diagonal_(0.0)
    contact = contact.unsqueeze(0).unsqueeze(0)

    if contact.shape[-1] != map_size:
        contact = F.interpolate(
            contact,
            size=(map_size, map_size),
            mode='nearest',
        )

    return contact.squeeze(0)


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    feature_channels = int(config.get('feature_channels', config.get('channels', 32)))
    residual_blocks = int(config.get('residual_blocks', 9))
    dropout_p = float(config.get('dropout_p', 0.25))
    map_size = int(config.get('map_size', 128))
    contact_cutoff = float(config.get('contact_cutoff', 8.0))

    model = ContactGANFingerprintNet(
        residual_blocks=residual_blocks,
        feature_channels=feature_channels,
        dropout_p=dropout_p,
    ).to(device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model, map_size, contact_cutoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--pdb-directory', default='pdbs')
    parser.add_argument('--dataset', default=DATASET_FILE)
    parser.add_argument('--output', default='umap_amp.png')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--n-neighbors', type=int, default=15)
    parser.add_argument('--min-dist', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    pdb_files = sorted(str(p) for p in Path(args.pdb_directory).glob('*.pdb'))
    if not pdb_files:
        raise FileNotFoundError(f'No PDB files found in {args.pdb_directory}')

    label_map = load_amp_labels(find_dataset(args.dataset))
    all_pdb_count = len(pdb_files)
    pdb_files = [p for p in pdb_files if Path(p).stem in label_map]
    print(f'PDB files in folder: {all_pdb_count}, matching the dataset: {len(pdb_files)}', flush=True)
    if not pdb_files:
        raise FileNotFoundError('No PDB file names match the dataset ids.')

    model, map_size, contact_cutoff = load_model(args.checkpoint, device)
    print(f'PDB files: {len(pdb_files)}', flush=True)
    print(f'Map size: {map_size}', flush=True)
    print(f'Contact cutoff: {contact_cutoff} A', flush=True)

    valid_paths = []
    maps = []

    for i, path in enumerate(pdb_files, start=1):
        coords = parse_ca_coordinates(path)
        if coords is None:
            continue

        maps.append(build_contact_map(coords, map_size, contact_cutoff))
        valid_paths.append(path)

        if i % 1000 == 0:
            print(f'Prepared {i}/{len(pdb_files)} PDB files', flush=True)

    if not maps:
        raise RuntimeError('No valid PDB structures were found.')

    fingerprints = []
    with torch.no_grad():
        for start in range(0, len(maps), args.batch_size):
            batch = torch.stack(maps[start:start + args.batch_size], dim=0).to(device)
            z = model.encode(batch)
            z = F.normalize(z, p=2, dim=1)
            fingerprints.append(z.cpu().numpy())

            done = min(start + args.batch_size, len(maps))
            if done % 1000 < args.batch_size or done == len(maps):
                print(f'Embedded {done}/{len(maps)} structures', flush=True)

    fingerprints = np.concatenate(fingerprints, axis=0)
    print(f'Fingerprint shape: {fingerprints.shape}', flush=True)

    ids = [Path(path).stem for path in valid_paths]
    labels = np.array([label_map[i] for i in ids], dtype=np.int8)

    print(
        f'AMP: {int(labels.sum())}, non-AMP: {int((labels == 0).sum())}, total: {len(labels)}',
        flush=True,
    )

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric='cosine',
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(fingerprints)

    anti_mask = labels == 1
    other_mask = labels == 0

    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding_2d[other_mask, 0],
        embedding_2d[other_mask, 1],
        c='#007AFF',
        label=f'Non-AMP (n={int(other_mask.sum())})',
        edgecolors='none',
        s=10,
        alpha=0.9,
        zorder=1,
    )
    plt.scatter(
        embedding_2d[anti_mask, 0],
        embedding_2d[anti_mask, 1],
        c='#FF3B30',
        label=f'AMP (n={int(anti_mask.sum())})',
        edgecolors='none',
        s=18,
        alpha=0.9,
        zorder=2,
    )
    plt.title('UMAP of Structural Fingerprints: AMP vs Non-AMP')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close()

    output_npz = str(Path(args.output).with_suffix('')) + '_embeddings.npz'
    np.savez_compressed(
        output_npz,
        names=np.array([os.path.basename(p) for p in valid_paths]),
        ids=np.array(ids),
        fingerprints=fingerprints,
        labels=labels,
        umap=embedding_2d,
    )

    print(f'Saved UMAP image: {args.output}', flush=True)
    print(f'Saved embeddings: {output_npz}', flush=True)


if __name__ == '__main__':
    main()
