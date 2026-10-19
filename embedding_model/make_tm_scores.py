from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

try:
    from tmtools import tm_align
except ImportError as error:
    raise SystemExit(
        "tmtools is required. Install it with: python -m pip install tmtools tqdm scikit-learn"
    ) from error


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z",
    "UNK": "X",
}

_WORKER_COORDS: list[np.ndarray] | None = None
_WORKER_SEQS: list[str] | None = None


def parse_longest_chain(path: Path) -> tuple[np.ndarray, str] | None:
    chains: dict[str, list[tuple[tuple[str, str], str, np.ndarray]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}

    try:
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.startswith("ENDMDL"):
                    break
                if not line.startswith("ATOM  "):
                    continue
                if line[12:16].strip() != "CA":
                    continue
                if line[16:17] not in (" ", "A"):
                    continue

                chain = line[21:22]
                residue_key = (line[22:26], line[26:27])
                seen.setdefault(chain, set())
                if residue_key in seen[chain]:
                    continue
                seen[chain].add(residue_key)

                residue = line[17:20].strip()
                xyz = np.asarray(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float64,
                )
                chains.setdefault(chain, []).append((residue_key, residue, xyz))
    except (OSError, ValueError, IndexError):
        return None

    if not chains:
        return None

    residues = max(chains.values(), key=len)
    if len(residues) < 4:
        return None

    coordinates = np.stack([entry[2] for entry in residues], axis=0)
    sequence = "".join(THREE_TO_ONE.get(entry[1], "X") for entry in residues)
    return np.ascontiguousarray(coordinates), sequence


def make_descriptor(
    coordinates: np.ndarray,
    descriptor_size: int,
    distance_clip: float,
) -> np.ndarray:
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.float64))
    distances = np.minimum(distances, distance_clip) / distance_clip

    n = distances.shape[0]
    sample_positions = np.linspace(0, n - 1, descriptor_size)
    sample_indices = np.clip(np.rint(sample_positions).astype(np.int64), 0, n - 1)
    sampled = distances[np.ix_(sample_indices, sample_indices)].astype(np.float32)

    # Add normalized length so similarly shaped peptides with very different lengths
    # do not become identical descriptors.
    length_feature = np.asarray([min(n / 256.0, 2.0)], dtype=np.float32)
    return np.concatenate([sampled.reshape(-1), length_feature], axis=0)


def build_candidate_pairs(
    reduced: np.ndarray,
    neighbor_candidates: int,
    random_candidates: int,
    seed: int,
) -> tuple[list[tuple[int, int]], int]:
    count = reduced.shape[0]
    rng = np.random.default_rng(seed)

    neighbor_count = min(neighbor_candidates + 1, count)
    nn = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean", n_jobs=-1)
    nn.fit(reduced)
    neighbor_indices = nn.kneighbors(return_distance=False)

    pair_set: set[tuple[int, int]] = set()
    directed_count = 0

    for i in tqdm(range(count), desc="Candidate graph", unit="structure"):
        chosen: set[int] = set()

        for j in neighbor_indices[i]:
            j = int(j)
            if j == i:
                continue
            chosen.add(j)
            if len(chosen) >= neighbor_candidates:
                break

        if random_candidates > 0:
            needed = random_candidates
            # Rejection sampling is cheap because the candidate set is tiny relative to N.
            while needed > 0:
                j = int(rng.integers(0, count))
                if j == i or j in chosen:
                    continue
                chosen.add(j)
                needed -= 1

        directed_count += len(chosen)
        for j in chosen:
            a, b = (i, j) if i < j else (j, i)
            pair_set.add((a, b))

    return sorted(pair_set), directed_count


def _init_worker(coords: list[np.ndarray], seqs: list[str]) -> None:
    global _WORKER_COORDS, _WORKER_SEQS
    _WORKER_COORDS = coords
    _WORKER_SEQS = seqs


def _align_pair(pair: tuple[int, int]) -> tuple[int, int, float, float, str | None]:
    i, j = pair
    try:
        assert _WORKER_COORDS is not None
        assert _WORKER_SEQS is not None
        result = tm_align(
            _WORKER_COORDS[i],
            _WORKER_COORDS[j],
            _WORKER_SEQS[i],
            _WORKER_SEQS[j],
        )
        score_ij = float(np.clip(result.tm_norm_chain1, 0.0, 1.0))
        score_ji = float(np.clip(result.tm_norm_chain2, 0.0, 1.0))
        return i, j, score_ij, score_ji, None
    except Exception as error:
        return i, j, float("nan"), float("nan"), repr(error)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-directory", default="pdbs")
    parser.add_argument("--output-directory", default="tm_scores")
    parser.add_argument("--neighbor-candidates", type=int, default=20)
    parser.add_argument("--random-candidates", type=int, default=10)
    parser.add_argument("--descriptor-size", type=int, default=32)
    parser.add_argument("--distance-clip", type=float, default=30.0)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument(
        "--max-new-alignments",
        type=int,
        default=0,
        help="Stop after this many new unique pair alignments. Use 0 for no limit.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1000,
        help="Flush scores.npy and done.npy after this many completed alignments.",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.neighbor_candidates < 0 or args.random_candidates < 0:
        raise SystemExit("Candidate counts cannot be negative.")

    pdb_directory = Path(args.pdb_directory)
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    pdb_paths = sorted(pdb_directory.glob("*.pdb"))
    if len(pdb_paths) < 2:
        raise SystemExit("At least two PDB files are required.")

    print(f"Parsing {len(pdb_paths)} PDB files...", flush=True)
    structures: list[tuple[Path, np.ndarray, str]] = []
    descriptors: list[np.ndarray] = []

    for path in tqdm(pdb_paths, unit="pdb"):
        parsed = parse_longest_chain(path)
        if parsed is None:
            continue
        coordinates, sequence = parsed
        structures.append((path, coordinates, sequence))
        descriptors.append(
            make_descriptor(coordinates, args.descriptor_size, args.distance_clip)
        )

    count = len(structures)
    if count < 2:
        raise SystemExit("Fewer than two valid structures were found.")

    names = [path.stem for path, _, _ in structures]
    if len(names) != len(set(names)):
        raise SystemExit("PDB filename stems must be unique.")

    descriptor_matrix = np.stack(descriptors, axis=0)
    pca_components = min(args.pca_components, descriptor_matrix.shape[1], count - 1)
    if pca_components < descriptor_matrix.shape[1]:
        print(f"Reducing descriptors to {pca_components} PCA dimensions...", flush=True)
        reduced = PCA(
            n_components=pca_components,
            random_state=args.seed,
            svd_solver="randomized",
        ).fit_transform(descriptor_matrix)
    else:
        reduced = descriptor_matrix

    print(
        f"Finding {args.neighbor_candidates} descriptor neighbors per structure...",
        flush=True,
    )
    candidate_pairs, directed_count = build_candidate_pairs(
        reduced,
        args.neighbor_candidates,
        args.random_candidates,
        args.seed,
    )

    names_path = output_directory / "names.json"
    scores_path = output_directory / "scores.npy"
    done_path = output_directory / "done.npy"
    candidate_path = output_directory / "candidate_pairs.npy"

    if names_path.exists():
        previous_names = json.loads(names_path.read_text())
        if previous_names != names:
            raise SystemExit(
                "The PDB set changed. Use a new output directory or delete the old TM-score files."
            )
    else:
        names_path.write_text(json.dumps(names, indent=2))

    # Store the unique candidate list for reproducibility and faster inspection.
    candidate_array = np.asarray(candidate_pairs, dtype=np.int32)
    np.save(candidate_path, candidate_array)

    if scores_path.exists() and done_path.exists():
        scores = np.lib.format.open_memmap(scores_path, mode="r+")
        done = np.lib.format.open_memmap(done_path, mode="r+")
        if scores.shape != (count, count) or done.shape != (count, count):
            raise SystemExit("The existing TM-score files have the wrong shape.")
    else:
        scores = np.lib.format.open_memmap(
            scores_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, count),
        )
        done = np.lib.format.open_memmap(
            done_path,
            mode="w+",
            dtype=np.uint8,
            shape=(count, count),
        )
        scores[:] = np.nan
        done[:] = 0
        np.fill_diagonal(scores, 1.0)
        np.fill_diagonal(done, 1)
        scores.flush()
        done.flush()

    pending_pairs = [(i, j) for i, j in candidate_pairs if not done[i, j]]
    already_complete = len(candidate_pairs) - len(pending_pairs)

    if args.max_new_alignments > 0:
        pending_pairs = pending_pairs[: args.max_new_alignments]

    print(f"Valid structures: {count:,}", flush=True)
    print(
        f"Candidates per structure: {args.neighbor_candidates + args.random_candidates}",
        flush=True,
    )
    print(f"Directed candidate scores: {directed_count:,}", flush=True)
    print(f"Unique candidate alignments: {len(candidate_pairs):,}", flush=True)
    print(f"Already complete: {already_complete:,}", flush=True)
    print(f"Remaining unique alignments: {len(pending_pairs):,}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    if not pending_pairs:
        print("All candidate alignments are complete.", flush=True)
        return

    coords = [entry[1] for entry in structures]
    seqs = [entry[2] for entry in structures]

    # Linux uses fork by default. Fork shares the parsed structures copy-on-write.
    # This avoids serializing all structures once per task.
    available_methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in available_methods else "spawn")

    completed_since_flush = 0
    failed = 0

    try:
        with context.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(coords, seqs),
        ) as pool:
            iterator = pool.imap_unordered(
                _align_pair,
                pending_pairs,
                chunksize=args.chunksize,
            )

            with tqdm(total=len(pending_pairs), unit="pair") as progress:
                for i, j, score_ij, score_ji, error in iterator:
                    if error is None:
                        scores[i, j] = score_ij
                        scores[j, i] = score_ji
                        done[i, j] = 1
                        done[j, i] = 1
                    else:
                        failed += 1

                    completed_since_flush += 1
                    progress.update(1)

                    if completed_since_flush >= args.flush_every:
                        scores.flush()
                        done.flush()
                        completed_since_flush = 0

    except KeyboardInterrupt:
        print("\nStopping workers and saving completed scores...", flush=True)
    finally:
        scores.flush()
        done.flush()

    if failed:
        print(
            f"Warning: {failed:,} alignments failed and remain incomplete for a later retry.",
            flush=True,
        )

    print(f"Saved TM-scores to {scores_path}", flush=True)
    print(f"Saved completion flags to {done_path}", flush=True)
    print(f"Saved candidate pairs to {candidate_path}", flush=True)


if __name__ == "__main__":
    main()
