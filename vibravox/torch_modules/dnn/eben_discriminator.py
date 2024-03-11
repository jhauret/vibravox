""" EBEN discriminators and sub blocks definition in Pytorch"""

import torch
from torch import nn

from vibravox.torch_modules.dnn.melgan_discriminator import DiscriminatorMelGAN
from vibravox.torch_modules.utils import normalized_conv1d


class DiscriminatorEBENMultiScales(nn.Module):
    """
    Multi-scales discriminators of composed of 3 scales pqmf discriminators refining q bands and one 1 full scale Melgan

    Args:
        q: The number of PMQF bands sent to the discriminators to be refined
    """

    def __init__(self, q: int = 3):
        super().__init__()

        self.q = q

        # PQMF discriminators
        self.pqmf_discriminators = torch.nn.ModuleList()

        # having multiple dilation helps to focus on multiscale structure of bands
        for dilation in [1, 2, 3]:
            self.pqmf_discriminators.append(DiscriminatorEBEN(dilation=dilation, q=q))

        # MelGAN discriminator
        self.melgan_discriminator = DiscriminatorMelGAN(alpha_leaky_relu=0.2)

    def forward(self, bands, audio):
        """
        Forward pass of the EBEN discriminators module.

        Args:
            bands (torch.Tensor): all PQMF bands
            audio (torch.Tensor): corresponding speech signal

        Returns:
            embeddings (List[List[torch.Tensor]]): a list of all embeddings layers of all discriminators
        """
        embeddings = []

        for dis in self.pqmf_discriminators:
            embeddings.append(dis(bands[:, -self.q:, :]))

        embeddings.append(self.melgan_discriminator(audio))

        return embeddings


class DiscriminatorEBEN(nn.Module):
    """
    EBEN PQMF-bands discriminator
    """

    def __init__(self, dilation=1, q: int = 3):
        super().__init__()

        self.dilation = dilation

        assert q in [1, 2, 3, 5, 6, 10, 15]

        self.discriminator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(1),
                    normalized_conv1d(
                        q,
                        30,
                        kernel_size=(3,),
                        stride=(1,),
                        padding=(1,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        30,
                        60,
                        kernel_size=(7,),
                        stride=(2,),
                        padding=(3,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        60,
                        120,
                        kernel_size=(7,),
                        stride=(2,),
                        padding=(3,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        120,
                        240,
                        kernel_size=(7,),
                        stride=(2,),
                        padding=(3,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        240,
                        480,
                        kernel_size=(7,),
                        stride=(2,),
                        padding=(3,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        480,
                        960,
                        kernel_size=(7,),
                        stride=(2,),
                        padding=(3,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        960,
                        960,
                        kernel_size=(5,),
                        stride=(1,),
                        padding=(2,),
                        dilation=self.dilation,
                        groups=q,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                normalized_conv1d(
                    960, 1, kernel_size=(3,), stride=(1,), padding=(1,), groups=1
                ),
            ]
        )

    def forward(self, bands):
        embeddings = [bands]
        for module in self.discriminator:
            embeddings.append(module(embeddings[-1]))
        return embeddings