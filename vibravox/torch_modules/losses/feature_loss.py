from typing import List

import torch
import torch.nn as nn


class FeatureLossForDiscriminatorMelganMultiScales(torch.nn.Module):
    """
    Feature loss as defined in [1].

    [1]: Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2022).
    High fidelity neural audio compression.
    arXiv preprint arXiv:2210.13438.
    """

    def __init__(self):
        super().__init__()

        self.l1 = nn.L1Loss()

    def forward(
        self,
        embeddings_a: List[List[torch.Tensor]],
        embeddings_b: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """

        Args:
            embeddings_a (List[List[torch.Tensor]]): Output of DiscriminatorMelganMultiScales on audio A.
                                                Usually this the enhanced speech
            embeddings_b (List[List[torch.Tensor]]): Output of DiscriminatorMelganMultiScales on audio B.
                                            Will be used to normalize the loss. Usually this the reference speech
        Returns:
            feature_loss (torch.Tensor): Feature loss value, is a scalar.
        """

        feature_loss = 0.0

        for scale_a, scale_b in zip(embeddings_a, embeddings_b):
            for layer_a, layer_b in zip(
                scale_a[1:-1], scale_b[1:-1]
            ):  # Do not consider audio and certainties
                feature_loss = feature_loss + self.l1(layer_a, layer_b) / torch.mean(
                    torch.abs(layer_a)
                )

        # Divide by number of layers and scales
        feature_loss = feature_loss / (len(embeddings_a) * len(scale_a[1:-1]))

        return feature_loss
