from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_NAME = "contactgan_resnet_deepfold_fingerprint_paper_loss_v1"


def symmetrize_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.size(-2) != tensor.size(-1):
        raise ValueError("The contact map must be square.")
    return 0.5 * (tensor + tensor.transpose(-1, -2))


def clear_diagonal(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.size(-2) != tensor.size(-1):
        raise ValueError("The contact map must be square.")
    size = tensor.size(-1)
    diagonal = torch.eye(size, dtype=tensor.dtype, device=tensor.device)
    diagonal = diagonal.view(1, 1, size, size)
    return tensor * (1.0 - diagonal)


class ContactGANResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int = 32,
        dilation: int = 1,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act1 = nn.PReLU(channels)
        self.dropout = nn.Dropout2d(dropout_p)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act2 = nn.PReLU(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv1(inputs)
        features = self.norm1(features)
        features = self.act1(features)
        features = self.dropout(features)
        features = self.conv2(features)
        features = self.norm2(features)
        return self.act2(inputs + features)


class ContactGANFingerprintNet(nn.Module):
    """One ContactGAN-style ResNet with a DeepFold-style fingerprint readout."""

    architecture_name = ARCHITECTURE_NAME

    def __init__(
        self,
        input_channels: int = 1,
        feature_channels: int = 32,
        residual_blocks: int = 9,
        dropout_p: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.residual_blocks = residual_blocks
        self.dropout_p = dropout_p
        self.embedding_dim = feature_channels

        self.input_layer = nn.Sequential(
            nn.Conv2d(
                input_channels,
                feature_channels,
                kernel_size=9,
                padding=4,
                bias=False,
            ),
            nn.InstanceNorm2d(feature_channels, affine=True),
            nn.PReLU(feature_channels),
        )

        dilation_cycle = (1, 2, 4)
        self.residual_trunk = nn.Sequential(
            *[
                ContactGANResidualBlock(
                    channels=feature_channels,
                    dilation=dilation_cycle[index % 3],
                    dropout_p=dropout_p,
                )
                for index in range(residual_blocks)
            ]
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def extract_feature_maps(self, contact_maps: torch.Tensor) -> torch.Tensor:
        if contact_maps.ndim != 4:
            raise ValueError("contact_maps must have shape [B, C, H, W].")
        if contact_maps.size(-2) != contact_maps.size(-1):
            raise ValueError("The contact map must be square.")
        return self.residual_trunk(self.input_layer(contact_maps))

    @staticmethod
    def make_raw_fingerprint(feature_maps: torch.Tensor) -> torch.Tensor:
        diagonal_values = torch.diagonal(feature_maps, dim1=-2, dim2=-1)
        return diagonal_values.mean(dim=-1)

    @classmethod
    def make_fingerprint(cls, feature_maps: torch.Tensor) -> torch.Tensor:
        raw = cls.make_raw_fingerprint(feature_maps)
        return F.normalize(raw, p=2, dim=1, eps=1e-8)

    def encode(self, contact_maps: torch.Tensor) -> torch.Tensor:
        return self.make_fingerprint(self.extract_feature_maps(contact_maps))

    def forward(
        self,
        contact_maps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.extract_feature_maps(contact_maps)
        fingerprint = self.make_fingerprint(feature_maps)
        reconstructed = self.output_layer(feature_maps)
        reconstructed = symmetrize_map(reconstructed)
        reconstructed = clear_diagonal(reconstructed)
        return reconstructed, fingerprint


class DeepFoldPaperRankingLoss(nn.Module):
    """DeepFold max-margin loss with online hard-pair mining.

    For each anchor, the caller supplies global positive and negative masks.
    The loss selects the least similar positive fingerprint in the batch and
    the top-k most similar negative fingerprints in the batch.
    """

    def __init__(self, margin: float = 0.1, hard_negative_count: int = 10) -> None:
        super().__init__()
        if margin < 0.0:
            raise ValueError("margin must be nonnegative.")
        if hard_negative_count < 1:
            raise ValueError("hard_negative_count must be positive.")
        self.margin = margin
        self.hard_negative_count = hard_negative_count

    def forward(
        self,
        fingerprints: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        if fingerprints.ndim != 2:
            raise ValueError("fingerprints must have shape [B, D].")
        batch_size = fingerprints.size(0)
        expected_shape = (batch_size, batch_size)
        if positive_mask.shape != expected_shape:
            raise ValueError("positive_mask must have shape [B, B].")
        if negative_mask.shape != expected_shape:
            raise ValueError("negative_mask must have shape [B, B].")

        fingerprints = F.normalize(fingerprints, p=2, dim=1, eps=1e-8)
        similarity = fingerprints @ fingerprints.transpose(0, 1)

        if anchor_mask is None:
            anchor_mask = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=fingerprints.device,
            )
        else:
            anchor_mask = anchor_mask.to(
                device=fingerprints.device,
                dtype=torch.bool,
            )

        positive_mask = positive_mask.to(
            device=fingerprints.device,
            dtype=torch.bool,
        )
        negative_mask = negative_mask.to(
            device=fingerprints.device,
            dtype=torch.bool,
        )

        anchor_losses: list[torch.Tensor] = []
        used_anchors = 0

        for anchor_index in range(batch_size):
            if not bool(anchor_mask[anchor_index]):
                continue

            positive_values = similarity[anchor_index][positive_mask[anchor_index]]
            negative_values = similarity[anchor_index][negative_mask[anchor_index]]

            if positive_values.numel() == 0 or negative_values.numel() == 0:
                continue

            hard_positive = positive_values.min()
            hard_negatives = negative_values.topk(
                k=min(self.hard_negative_count, negative_values.numel()),
                largest=True,
            ).values

            # DeepFold Equation 3 applied to the mined hard positive and
            # hard negatives. Sum over negatives, then average over anchors.
            anchor_loss = F.relu(
                hard_negatives - hard_positive + self.margin
            ).sum()
            anchor_losses.append(anchor_loss)
            used_anchors += 1

        if not anchor_losses:
            return fingerprints.sum() * 0.0, 0

        return torch.stack(anchor_losses).mean(), used_anchors


class ContactGANDeepFoldLoss(nn.Module):
    """Contact reconstruction plus the paper-faithful DeepFold ranking loss."""

    def __init__(
        self,
        fingerprint_weight: float = 1.0,
        margin: float = 0.1,
        hard_negative_count: int = 10,
    ) -> None:
        super().__init__()
        if fingerprint_weight < 0.0:
            raise ValueError("fingerprint_weight must be nonnegative.")
        self.fingerprint_weight = fingerprint_weight
        self.reconstruction_loss = nn.MSELoss()
        self.ranking_loss = DeepFoldPaperRankingLoss(
            margin=margin,
            hard_negative_count=hard_negative_count,
        )

    def forward(
        self,
        reconstructed_maps: torch.Tensor,
        target_maps: torch.Tensor,
        fingerprints: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        anchor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        reconstruction = self.reconstruction_loss(
            reconstructed_maps.clamp(0.0, 1.0),
            target_maps.clamp(0.0, 1.0),
        )
        fingerprint, used_anchors = self.ranking_loss(
            fingerprints,
            positive_mask,
            negative_mask,
            anchor_mask,
        )
        total = reconstruction + self.fingerprint_weight * fingerprint
        return total, reconstruction, fingerprint, used_anchors
