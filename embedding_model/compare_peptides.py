import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tmtools import tm_align

if __package__:
    from .models import ContactGANFingerprintNet, clear_diagonal, symmetrize_map
    from .train import parse_contact_coordinates
    from .make_tm_scores import parse_longest_chain
else:
    from models import ContactGANFingerprintNet, clear_diagonal, symmetrize_map
    from train import parse_contact_coordinates
    from make_tm_scores import parse_longest_chain


def compare_peptides(checkpoint_path, pdb1, pdb2, device="cpu"):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_config = checkpoint["model_config"]
    training_config = checkpoint["training_config"]

    model = ContactGANFingerprintNet(
        input_channels=model_config["input_channels"],
        feature_channels=model_config["feature_channels"],
        residual_blocks=model_config["residual_blocks"],
        dropout_p=model_config["dropout_p"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    map_size = training_config["map_size"]
    cutoff = training_config["contact_cutoff_angstrom"]

    contact_maps = []
    structures = []

    for pdb_path in (pdb1, pdb2):
        parsed_contacts = parse_contact_coordinates(str(pdb_path))

        parsed_structure = parse_longest_chain(Path(pdb_path))

        if parsed_contacts is None or parsed_structure is None:
            raise ValueError(f"Could not parse peptide structure: {pdb_path}")

        _, coordinates = parsed_contacts

        distances = torch.cdist(coordinates, coordinates, p=2)
        contact_map = (distances <= cutoff).float()
        contact_map.fill_diagonal_(0.0)

        contact_map = F.interpolate(
            contact_map[None, None],
            size=(map_size, map_size),
            mode="nearest",
        )
        contact_map = clear_diagonal(symmetrize_map(contact_map))

        contact_maps.append(contact_map)
        structures.append(parsed_structure)

    with torch.inference_mode():
        batch = torch.cat(contact_maps, dim=0).to(device)
        embeddings = model.encode(batch)

        cosine = F.cosine_similarity(
            embeddings[0:1],
            embeddings[1:2],
            dim=1,
        ).item()

    coordinates1, sequence1 = structures[0]
    coordinates2, sequence2 = structures[1]

    alignment = tm_align(
        coordinates1,
        coordinates2,
        sequence1,
        sequence2,
    )

    score1 = float(alignment.tm_norm_chain1)
    score2 = float(alignment.tm_norm_chain2)

    return {
        "cosine_similarity": cosine,
        "tm_score_chain1": score1,
        "tm_score_chain2": score2,
        "tm_score": (score1 + score2) / 2.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate peptide embeddings and compare two PDB structures."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("pdb1")
    parser.add_argument("pdb2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    scores = compare_peptides(
        args.checkpoint,
        args.pdb1,
        args.pdb2,
        device=args.device,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
