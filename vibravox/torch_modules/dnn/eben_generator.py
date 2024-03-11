""" EBEN generator and sub blocks definition in Pytorch"""

import torch
from torch import nn

from vibravox.torch_modules.dsp.pqmf import PseudoQMFBanks
from vibravox.torch_modules.utils import normalized_conv1d, normalized_conv_trans1d


class EBENGenerator(nn.Module):
    def __init__(self, m: int, n: int, p: int):
        """
        Generator of EBEN

        Args:
            m: The number of PQMF bands, which is also the decimation factor of the waveform after the analysis step
            p: The number of informative PMQF bands sent to the generator
            n: The kernel size of PQMF
        """
        super().__init__()

        self.p = p
        self.pqmf = PseudoQMFBanks(decimation=m, kernel_size=n)

        # multiple is the product of encoder_blocks strides and PQMF decimation
        self.multiple = 2 * 4 * 8 * m

        self.nl = nn.LeakyReLU(negative_slope=0.01)

        self.first_conv = nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncBlock(out_channels=64, stride=2, nl=self.nl),
                EncBlock(out_channels=128, stride=4, nl=self.nl),
                EncBlock(out_channels=256, stride=8, nl=self.nl),
            ]
        )

        self.latent_conv = nn.Sequential(
            self.nl,
            normalized_conv1d(
                in_channels=256,
                out_channels=64,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            self.nl,
            normalized_conv1d(
                in_channels=64,
                out_channels=256,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
            ),
            self.nl,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecBlock(out_channels=128, stride=8, nl=self.nl),
                DecBlock(out_channels=64, stride=4, nl=self.nl),
                DecBlock(out_channels=32, stride=2, nl=self.nl),
            ]
        )

        self.last_conv = nn.Conv1d(
            in_channels=32,
            out_channels=4,
            kernel_size=3,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, cut_audio):
        """
        Forward pass of the EBEN generator module.

        Args:
            cut_audio (torch.Tensor): in-ear speech signal

        Returns:
            enhanced_speech (torch.Tensor): The enhanced signal.
            enhanced_speech_decomposed (torch.Tensor): The enhanced PQMF bands before the synthesis stage
        """

        # PQMF analysis, for P first bands only
        first_bands = self.pqmf(cut_audio, "analysis", bands=self.p)

        # First conv
        x = self.first_conv(first_bands)

        # Encoder forward
        x1 = self.encoder_blocks[0](self.nl(x))
        x2 = self.encoder_blocks[1](self.nl(x1))
        x3 = self.encoder_blocks[2](self.nl(x2))

        # Latent forward
        x = self.latent_conv(x3)

        # Decoder forward
        x = self.decoder_blocks[0](x, x3)
        x = self.decoder_blocks[1](x, x2)
        x = self.decoder_blocks[2](x, x1)

        # Last conv
        x = self.last_conv(x)

        # Recompose PQMF bands ( + avoiding any inplace operation for backprop )
        b, c, t = first_bands.shape  # (batch_size, channels, time_len)
        fill_up_tensor = torch.zeros(
            (b, self.pqmf.decimation - self.p, t), requires_grad=False
        ).type_as(first_bands)
        cat_tensor = torch.cat(tensors=(first_bands, fill_up_tensor), dim=1)
        enhanced_speech_decomposed = torch.tanh(x + cat_tensor)
        enhanced_speech = torch.sum(
            self.pqmf(enhanced_speech_decomposed, "synthesis"), 1, keepdim=True
        )

        return enhanced_speech, enhanced_speech_decomposed

    def cut_to_valid_length(self, tensor):
        """This function is used to make a tensor divisible by the minimal chunk length"""

        old_len = tensor.shape[2]
        new_len = old_len - (old_len + self.pqmf.kernel_size) % self.multiple
        tensor = torch.narrow(tensor, 2, 0, new_len)

        return tensor


class DecBlock(nn.Module):
    """
    Decoder Block Module
    """

    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels, nl=nl, dilation=9),
        )

        self.conv_trans = normalized_conv_trans1d(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride // 2,
            output_padding=0,
            bias=bias,
        )

    def forward(self, x, encoder_output):
        x = x + encoder_output
        out = self.residuals(self.nl(self.conv_trans(x)))
        return out


class EncBlock(nn.Module):
    """
    Encoder Block Module
    """

    def __init__(self, out_channels, stride, nl, bias=False):
        super().__init__()

        self.nl = nl

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=1),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=3),
            ResidualUnit(channels=out_channels // 2, nl=nl, dilation=9),
        )
        self.conv = normalized_conv1d(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride - 1,
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, x):
        out = self.conv(self.residuals(x))
        return out


class ResidualUnit(nn.Module):
    """
    Residual Unit Module
    """

    def __init__(self, channels, nl, dilation, bias=False):
        super().__init__()

        self.dilated_conv = normalized_conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.pointwise_conv = normalized_conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=bias,
            padding_mode="reflect",
        )
        self.nl = nl

    def forward(self, x):
        out = x + self.nl(self.pointwise_conv(self.dilated_conv(x)))
        return out
