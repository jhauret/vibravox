from typing import List

import torch


class HingeLossForDiscriminatorMelganMultiScales(torch.nn.Module):
    """
    Hinge loss as defined in [1].

    [1]: Kumar, K., Kumar, R., De Boissiere, T., Gestin, L., Teoh, W. Z., Sotelo, J., ... & Courville, A. C. (2019).
     Melgan: Generative adversarial networks for conditional waveform synthesis.
      Advances in neural information processing systems, 32.
    """

    def __init__(self):
        super().__init__()

        self.relu = torch.nn.ReLU()

    def forward(
        self,
        embeddings: List[List[torch.Tensor]],
        target: float,
    ):
        """

        Args:
            embeddings (List[List[Tensor]]): Output of DiscriminatorMelganMultiScales. Tensor shape: (batch, channel, time)
            target (float): target label for discriminator certainties, should be -1 for fake and 1 for real

        Returns:
            hinge_loss (Tensor): Hinge loss value, is a scalar.
        """

        hinge_loss = 0.0

        for scale_embedding in embeddings:
            certainties = scale_embedding[-1]
            hinge_loss = (
                hinge_loss + self.relu(1 - target * certainties).mean()
            )  # across time

        return hinge_loss / len(embeddings)
