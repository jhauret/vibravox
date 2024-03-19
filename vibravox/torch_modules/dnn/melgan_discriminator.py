"""
Implementation of the MelGAN [1] discriminator modules

[1]: Kumar, K., Kumar, R., De Boissiere, T., Gestin, L., Teoh, W. Z., Sotelo, J., ... & Courville, A. C. (2019).
 Melgan: Generative adversarial networks for conditional waveform synthesis.
  Advances in neural information processing systems, 32.
"""


import torch
import torch.nn as nn
from torchaudio.transforms import Resample

from vibravox.torch_modules.utils import normalized_conv1d


class MelganMultiScalesDiscriminator(nn.Module):
    def __init__(
        self, sample_rate: int, scales: int = 3, alpha_leaky_relu: float = 0.2
    ):
        """
        Aggregation of multiple MelGAN discriminators to extract features at multiple scales

        Args:
            sample_rate (int): Sample rate of the input audio
            scales (int): Number of scales to use
            alpha_leaky_relu (float): Slope of the negative part of the LeakyReLU activation function
        """

        super().__init__()

        self.discriminators = torch.nn.ModuleList()
        self.downsamplers = torch.nn.ModuleList()

        for scale in range(scales):
            self.discriminators.append(DiscriminatorMelGAN(alpha_leaky_relu))
            self.downsamplers.append(
                Resample(
                    orig_freq=sample_rate,
                    resampling_method="sinc_interp_kaiser",
                    new_freq=sample_rate // 2 ** scale,
                )
            )

    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio (Tensor): Input waveform (batches, 1, samples)

        Returns:
            (List[List[Tensor]]): List of scales of discriminator embeddings at each layer. Tensor shape: (batch, channel, time)
        """
        scales_embeddings = list()
        downsampled_versions = self.get_downsampled_versions(audio)

        for scale, signal in enumerate(downsampled_versions):
            scales_embeddings.append(self.discriminators[scale](signal))

        return scales_embeddings

    def get_downsampled_versions(self, audio: torch.Tensor):
        """
        Get downsampled versions of the input audio

        Args:
            audio (Tensor): Input waveform (batches, 1, samples)

        Returns:
            (List[Tensor]): List of downsampled versions of the input audio
        """
        downsampled_versions = [downsampler(audio) for downsampler in self.downsamplers]

        return downsampled_versions


class DiscriminatorMelGAN(nn.Module):
    def __init__(self, alpha_leaky_relu: float):
        """
        MelGAN discriminator
        # implementation inspired from https://github.com/seungwonpark/melgan/blob/master/model/discriminator.py

        BSD 3-Clause License

        Args:
            alpha_leaky_relu (float): Slope of the negative part of the LeakyReLU activation function
        """
        super().__init__()

        self.discriminator = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReflectionPad1d(7),
                    normalized_conv1d(
                        in_channels=1, out_channels=16, kernel_size=(15,), stride=(1,)
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        in_channels=16,
                        out_channels=64,
                        kernel_size=(41,),
                        stride=(4,),
                        padding=20,
                        groups=4,
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        in_channels=64,
                        out_channels=256,
                        kernel_size=(41,),
                        stride=(4,),
                        padding=20,
                        groups=4,
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        in_channels=256,
                        out_channels=1024,
                        kernel_size=(41,),
                        stride=(4,),
                        padding=20,
                        groups=4,
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=(41,),
                        stride=(4,),
                        padding=20,
                        groups=4,
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                nn.Sequential(
                    normalized_conv1d(
                        in_channels=1024,
                        out_channels=1024,
                        kernel_size=(5,),
                        stride=(1,),
                        padding=2,
                    ),
                    nn.LeakyReLU(alpha_leaky_relu, inplace=True),
                ),
                normalized_conv1d(
                    in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=1
                ),
            ]
        )

    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio (Tensor): Input waveform (batches, 1, samples)

        Returns:
            (List[Tensor]): List of discriminator embeddings at each layer. Tensor shape: (batch, channel, time)
        """
        embeddings = [audio]
        for module in self.discriminator:
            embeddings.append(module(embeddings[-1]))
        return embeddings
