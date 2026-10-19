from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler

from models import ContactGANDeepFoldLoss, ContactGANFingerprintNet, clear_diagonal, symmetrize_map


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PDB_DIRECTORY = os.environ.get("PDB_DIRECTORY", "pdbs")
TM_SCORE_DIRECTORY = os.environ.get("TM_SCORE_DIRECTORY", "tm_scores")
MAP_SIZE = int(os.environ.get("MAP_SIZE", "128"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
FINGERPRINT_WEIGHT = float(os.environ.get("FINGERPRINT_WEIGHT", "1.0"))
DEEPFOLD_MARGIN = float(os.environ.get("DEEPFOLD_MARGIN", "0.1"))
POSITIVE_FRACTION = float(os.environ.get("POSITIVE_FRACTION", "0.9"))
HARD_NEGATIVE_COUNT = int(os.environ.get("HARD_NEGATIVE_COUNT", "10"))
CONTACT_CUTOFF_ANGSTROM = float(os.environ.get("CONTACT_CUTOFF_ANGSTROM", "8.0"))
PRINT_INTERVAL = int(os.environ.get("PRINT_INTERVAL", "10"))
CHECKPOINT_DIRECTORY = os.environ.get("CHECKPOINT_DIRECTORY", "checkpoints_contactgan_deepfold")
CHECKPOINT_INTERVAL = int(os.environ.get("CHECKPOINT_INTERVAL", "10"))
RESUME_CHECKPOINT = os.environ.get("RESUME_CHECKPOINT")

FEATURE_CHANNELS = 32
RESIDUAL_BLOCKS = 9
DROPOUT_P = 0.25

if BATCH_SIZE < 4 or BATCH_SIZE % 2 != 0:
    raise ValueError("BATCH_SIZE must be an even integer of at least 4.")


@dataclass(frozen=True)
class ContactMapSample:
    dataset_index: int
    name: str
    contact_map: torch.Tensor


def parse_contact_coordinates(path: str) -> tuple[str, torch.Tensor] | None:
    chains: dict[bytes, dict[tuple[bytes, bytes], dict[bytes, tuple[float, float, float]]]] = {}
    order: dict[bytes, list[tuple[bytes, bytes]]] = {}

    try:
        with open(path, "rb") as handle:
            for line in handle:
                if line.startswith(b"ENDMDL"):
                    break
                if not line.startswith((b"ATOM  ", b"HETATM")):
                    continue
                if line[16:17] not in (b" ", b"A"):
                    continue
                atom = line[12:16].strip()
                if atom not in (b"CA", b"CB"):
                    continue

                chain = line[21:22]
                residue = (line[22:26], line[26:27])
                chains.setdefault(chain, {})
                order.setdefault(chain, [])
                if residue not in chains[chain]:
                    chains[chain][residue] = {}
                    order[chain].append(residue)
                chains[chain][residue][atom] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
    except (OSError, ValueError, IndexError):
        return None

    candidates: list[list[tuple[float, float, float]]] = []
    for chain, residues in order.items():
        coordinates: list[tuple[float, float, float]] = []
        for residue in residues:
            atoms = chains[chain][residue]
            ca = atoms.get(b"CA")
            if ca is None:
                continue
            coordinates.append(atoms.get(b"CB", ca))
        if len(coordinates) >= 4:
            candidates.append(coordinates)

    if not candidates:
        return None
    coordinates = max(candidates, key=len)
    return Path(path).stem, torch.tensor(coordinates, dtype=torch.float32)


def coordinates_to_contact_map(coordinates: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(coordinates, coordinates, p=2)
    contact_map = (distances <= CONTACT_CUTOFF_ANGSTROM).float()
    contact_map.fill_diagonal_(0.0)
    contact_map = F.interpolate(
        contact_map[None, None],
        size=(MAP_SIZE, MAP_SIZE),
        mode="nearest",
    )
    return clear_diagonal(symmetrize_map(contact_map)).squeeze(0)


class ContactDataset(Dataset[ContactMapSample]):
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = list(paths)
        self.names = [Path(path).stem for path in self.paths]
        self.cache: dict[int, ContactMapSample] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ContactMapSample:
        if index in self.cache:
            return self.cache[index]
        parsed = parse_contact_coordinates(self.paths[index])
        if parsed is None:
            raise RuntimeError(f"Could not parse {self.paths[index]}")
        name, coordinates = parsed
        sample = ContactMapSample(index, name, coordinates_to_contact_map(coordinates))
        self.cache[index] = sample
        return sample


def collate_samples(batch: list[ContactMapSample]) -> list[ContactMapSample]:
    return batch


class DeepFoldPairBatchSampler(Sampler[list[int]]):
    """Create batches with one known global positive for each anchor.

    The first half contains anchors. The second half contains one positive
    sampled for each corresponding anchor.
    """

    def __init__(
        self,
        positive_indices: Sequence[np.ndarray],
        batch_size: int,
        seed: int,
    ) -> None:
        self.positive_indices = list(positive_indices)
        self.batch_size = batch_size
        self.anchor_count = batch_size // 2
        self.seed = seed
        self.epoch = 0
        self.valid_anchors = np.asarray(
            [i for i, positives in enumerate(self.positive_indices) if len(positives) > 0],
            dtype=np.int64,
        )
        if len(self.valid_anchors) < self.anchor_count:
            raise RuntimeError("Too few structures have a positive TM-score neighbor.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return int(np.ceil(len(self.valid_anchors) / self.anchor_count))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        anchors = self.valid_anchors.copy()
        rng.shuffle(anchors)

        for start in range(0, len(anchors), self.anchor_count):
            anchor_batch = anchors[start:start + self.anchor_count]
            if len(anchor_batch) < 2:
                continue
            positive_batch = [
                int(rng.choice(self.positive_indices[int(anchor)]))
                for anchor in anchor_batch
            ]
            yield [int(value) for value in anchor_batch] + positive_batch


@dataclass(frozen=True)
class TMLabels:
    names: list[str]
    scores: np.ndarray
    done: np.ndarray
    name_to_index: dict[str, int]


def load_tm_labels() -> TMLabels:
    directory = Path(TM_SCORE_DIRECTORY)
    names_path = directory / "names.json"
    scores_path = directory / "scores.npy"
    done_path = directory / "done.npy"

    if not names_path.exists() or not scores_path.exists() or not done_path.exists():
        raise FileNotFoundError(
            "TM-score files are missing. Run make_tm_scores.py before training."
        )

    names = json.loads(names_path.read_text())
    scores = np.load(scores_path, mmap_mode="r")
    done = np.load(done_path, mmap_mode="r")

    expected_shape = (len(names), len(names))
    if scores.shape != expected_shape:
        raise ValueError(
            f"scores.npy has shape {scores.shape}, expected {expected_shape}."
        )
    if done.shape != expected_shape:
        raise ValueError(
            f"done.npy has shape {done.shape}, expected {expected_shape}."
        )

    known_directed = int(np.count_nonzero(done))
    print(
        f"Loaded sparse TM-score data: {known_directed:,} known directed entries ",
        f"for {len(names):,} structures.",
        flush=True,
    )

    return TMLabels(
        names=names,
        scores=scores,
        done=done,
        name_to_index={name: i for i, name in enumerate(names)},
    )


@dataclass(frozen=True)
class PositiveGraph:
    lists: list[np.ndarray]
    sets: list[set[int]]


def build_positive_graph(dataset: ContactDataset, labels: TMLabels) -> PositiveGraph:
    missing = [name for name in dataset.names if name not in labels.name_to_index]
    if missing:
        raise RuntimeError(f"TM-scores are missing for {len(missing)} structures.")

    global_indices = np.asarray(
        [labels.name_to_index[name] for name in dataset.names],
        dtype=np.int64,
    )
    positive_lists: list[np.ndarray] = []
    positive_sets: list[set[int]] = []

    print(
        f"Building DeepFold positive sets from sparse TM-score candidates for {len(dataset)} structures...",
        flush=True,
    )
    anchors_with_scores = 0
    anchors_with_positives = 0

    for local_i, global_i in enumerate(global_indices):
        row = np.asarray(
            labels.scores[int(global_i), global_indices],
            dtype=np.float32,
        ).copy()
        known = np.asarray(
            labels.done[int(global_i), global_indices],
            dtype=bool,
        ).copy()

        known[local_i] = False
        row[~known] = np.nan

        known_indices = np.flatnonzero(known)
        if known_indices.size == 0:
            positives = np.empty(0, dtype=np.int64)
        else:
            anchors_with_scores += 1
            tm_max = float(np.nanmax(row))
            threshold = POSITIVE_FRACTION * tm_max
            positives = np.flatnonzero(known & (row >= threshold)).astype(np.int64)

        if positives.size > 0:
            anchors_with_positives += 1

        positive_lists.append(positives)
        positive_sets.append(set(int(value) for value in positives))

    print(
        f"Anchors with known candidate scores: {anchors_with_scores:,}/{len(dataset):,}",
        flush=True,
    )
    print(
        f"Anchors with at least one DeepFold positive: {anchors_with_positives:,}/{len(dataset):,}",
        flush=True,
    )

    return PositiveGraph(positive_lists, positive_sets)


def batch_masks(
    samples: Sequence[ContactMapSample],
    positive_graph: PositiveGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(samples)
    positive = np.zeros((count, count), dtype=bool)
    for row_index, anchor in enumerate(samples):
        anchor_positives = positive_graph.sets[anchor.dataset_index]
        for column_index, candidate in enumerate(samples):
            if row_index == column_index:
                continue
            positive[row_index, column_index] = candidate.dataset_index in anchor_positives

    diagonal = np.eye(count, dtype=bool)
    negative = ~positive
    negative[diagonal] = False

    anchor_mask = np.zeros(count, dtype=bool)
    anchor_mask[: count // 2] = True

    return (
        torch.from_numpy(positive).to(DEVICE),
        torch.from_numpy(negative).to(DEVICE),
        torch.from_numpy(anchor_mask).to(DEVICE),
    )


def corrupt_contact_maps(targets: torch.Tensor) -> torch.Tensor:
    if not targets.numel():
        return targets
    noise = torch.rand_like(targets)
    dropped = torch.where((targets > 0.5) & (noise < 0.05), torch.zeros_like(targets), targets)
    false_mask = (targets <= 0.5) & (torch.rand_like(targets) < 0.005)
    corrupted = torch.where(false_mask, torch.ones_like(dropped), dropped)
    corrupted = clear_diagonal(symmetrize_map(corrupted))
    return corrupted


def save_checkpoint(
    path: str,
    epoch: int,
    model: ContactGANFingerprintNet,
    optimizer: torch.optim.Optimizer,
    best_validation: float,
    metrics: dict[str, float],
) -> None:
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "architecture_name": model.architecture_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_validation,
        "metrics": metrics,
        "model_config": {
            "input_channels": 1,
            "feature_channels": FEATURE_CHANNELS,
            "residual_blocks": RESIDUAL_BLOCKS,
            "dropout_p": DROPOUT_P,
            "embedding_dim": FEATURE_CHANNELS,
        },
        "training_config": {
            "map_size": MAP_SIZE,
            "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
            "deepfold_margin": DEEPFOLD_MARGIN,
            "positive_fraction": POSITIVE_FRACTION,
            "hard_negative_count": HARD_NEGATIVE_COUNT,
            "fingerprint_weight": FINGERPRINT_WEIGHT,
            "tm_score_mode": "global_precomputed_deepfold",
        },
    }
    temporary = f"{path}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    model: ContactGANFingerprintNet,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float]:
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"]), float(checkpoint.get("best_validation_loss", float("inf")))


def run_epoch(
    model: ContactGANFingerprintNet,
    criterion: ContactGANDeepFoldLoss,
    loader: DataLoader[list[ContactMapSample]],
    positive_graph: PositiveGraph,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "recon": 0.0, "fingerprint": 0.0}
    sample_count = 0
    used_anchor_count = 0

    for batch_index, samples in enumerate(loader, start=1):
        targets = torch.stack([sample.contact_map for sample in samples]).to(DEVICE)
        inputs = corrupt_contact_maps(targets) if training else targets
        positive_mask, negative_mask, anchor_mask = batch_masks(samples, positive_graph)

        if training:
            optimizer.zero_grad(set_to_none=True)

        reconstructed, fingerprints = model(inputs)
        total, recon, fingerprint, used_anchors = criterion(
            reconstructed,
            targets,
            fingerprints,
            positive_mask,
            negative_mask,
            anchor_mask,
        )

        if training:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        batch_size = len(samples)
        sample_count += batch_size
        used_anchor_count += used_anchors
        sums["total"] += float(total.item()) * batch_size
        sums["recon"] += float(recon.item()) * batch_size
        sums["fingerprint"] += float(fingerprint.item()) * batch_size

        if training and (
            batch_index == 1
            or batch_index % PRINT_INTERVAL == 0
            or batch_index == len(loader)
        ):
            with torch.no_grad():
                similarity = fingerprints @ fingerprints.T
                off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=DEVICE)
                cos_std = float(similarity[off_diagonal].std().item())
            print(
                f"Epoch {epoch:03d} | batch {batch_index:04d}/{len(loader):04d} | "
                f"Recon {recon.item():.6f} | Fingerprint {fingerprint.item():.6f} | "
                f"Anchors {used_anchors:02d} | CosStd {cos_std:.4f}",
                flush=True,
            )

    if sample_count == 0:
        raise RuntimeError("No samples were processed.")
    return {
        "total": sums["total"] / sample_count,
        "recon": sums["recon"] / sample_count,
        "fingerprint": sums["fingerprint"] / sample_count,
        "used_anchors": float(used_anchor_count),
    }


def make_loader(
    dataset: ContactDataset,
    positive_graph: PositiveGraph,
    seed: int,
) -> tuple[DataLoader[list[ContactMapSample]], DeepFoldPairBatchSampler]:
    sampler = DeepFoldPairBatchSampler(positive_graph.lists, BATCH_SIZE, seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_samples,
        persistent_workers=NUM_WORKERS > 0,
    )
    return loader, sampler


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    labels = load_tm_labels()
    pdb_files = sorted(glob.glob(os.path.join(PDB_DIRECTORY, "*.pdb")))
    pdb_files = [path for path in pdb_files if Path(path).stem in labels.name_to_index]
    if len(pdb_files) < 10:
        raise RuntimeError("Too few PDB files match the TM-score dataset.")

    train_files, validation_files = train_test_split(
        pdb_files,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    train_dataset = ContactDataset(train_files)
    validation_dataset = ContactDataset(validation_files)

    train_positive_graph = build_positive_graph(train_dataset, labels)
    validation_positive_graph = build_positive_graph(validation_dataset, labels)

    train_loader, train_sampler = make_loader(train_dataset, train_positive_graph, RANDOM_SEED)
    validation_loader, validation_sampler = make_loader(
        validation_dataset,
        validation_positive_graph,
        RANDOM_SEED + 100000,
    )

    model = ContactGANFingerprintNet(
        input_channels=1,
        feature_channels=FEATURE_CHANNELS,
        residual_blocks=RESIDUAL_BLOCKS,
        dropout_p=DROPOUT_P,
    ).to(DEVICE)
    criterion = ContactGANDeepFoldLoss(
        fingerprint_weight=FINGERPRINT_WEIGHT,
        margin=DEEPFOLD_MARGIN,
        hard_negative_count=HARD_NEGATIVE_COUNT,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    start_epoch = 0
    best_validation = float("inf")
    if RESUME_CHECKPOINT:
        start_epoch, best_validation = load_checkpoint(RESUME_CHECKPOINT, model, optimizer)

    print(f"Device: {DEVICE}", flush=True)
    print(f"Training structures: {len(train_dataset)}", flush=True)
    print(f"Validation structures: {len(validation_dataset)}", flush=True)
    print(f"Batch size: {BATCH_SIZE} = {BATCH_SIZE // 2} anchors + one positive each", flush=True)
    print("Fingerprint labels: global TM-score positives from 0.9 x TMmax.", flush=True)
    print(f"Hard negatives: top {HARD_NEGATIVE_COUNT} by fingerprint cosine similarity", flush=True)
    print(f"DeepFold margin: {DEEPFOLD_MARGIN}", flush=True)

    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)

    for epoch_index in range(start_epoch, EPOCHS):
        epoch = epoch_index + 1
        train_sampler.set_epoch(epoch)
        validation_sampler.set_epoch(epoch)
        print(f"\nStarting epoch {epoch}/{EPOCHS}", flush=True)

        train_metrics = run_epoch(
            model,
            criterion,
            train_loader,
            train_positive_graph,
            optimizer,
            epoch,
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model,
                criterion,
                validation_loader,
                validation_positive_graph,
                None,
                epoch,
            )

        metrics = {
            "train_total": train_metrics["total"],
            "train_recon": train_metrics["recon"],
            "train_fingerprint": train_metrics["fingerprint"],
            "validation_total": validation_metrics["total"],
            "validation_recon": validation_metrics["recon"],
            "validation_fingerprint": validation_metrics["fingerprint"],
        }

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | Total {train_metrics['total']:.6f} | "
            f"Recon {train_metrics['recon']:.6f} | Fingerprint {train_metrics['fingerprint']:.6f} | "
            f"ValTotal {validation_metrics['total']:.6f} | "
            f"ValRecon {validation_metrics['recon']:.6f} | "
            f"ValFingerprint {validation_metrics['fingerprint']:.6f}",
            flush=True,
        )

        latest = os.path.join(CHECKPOINT_DIRECTORY, "latest.pt")
        save_checkpoint(latest, epoch, model, optimizer, best_validation, metrics)

        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            save_checkpoint(
                os.path.join(CHECKPOINT_DIRECTORY, "best.pt"),
                epoch,
                model,
                optimizer,
                best_validation,
                metrics,
            )

        if epoch % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                os.path.join(CHECKPOINT_DIRECTORY, f"epoch_{epoch:03d}.pt"),
                epoch,
                model,
                optimizer,
                best_validation,
                metrics,
            )


if __name__ == "__main__":
    main()

