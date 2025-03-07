import torch
from torch import nn
import torch.nn.functional as F

def normalized_conv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))


def normalized_conv_trans1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        """
        Causal convolution that also supports stateful buffers for streaming inference.

        usage:
            # Training loop (full sequences)
            model.train()
            model.conv.set_streaming(False)  # Disable streaming during training

            # Switch to streaming mode
            model.eval()
            model.conv.set_streaming(True)
            model.conv.reset_state()  # Reset state for a new sequence

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of the convolution.
            stride (int): Stride of the convolution.
            dilation (int): Dilation factor of the convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to include a bias term in the convolution.
        """
        super().__init__()

        self.padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias
        )

        # State buffer for streaming (inference)
        self.register_buffer('state', torch.zeros(4, in_channels, self.padding))
        self.training_mode = True  # Default to training mode

    def forward(self, x):
        if self.streaming:
            # Use state for streaming inference
            x = torch.cat((self.state, x), dim=-1)
            out = self.conv(x)
            if self.padding!=0:
                self.state = x[:, :, -self.padding:].detach()

        else:
            # Apply causal padding directly during training
            x_padded = F.pad(x, (self.padding, 0))
            out = self.conv(x_padded)
            if self.padding!=0:
                out = out[:, :, :-self.padding]  # Remove causal padding

        return out

    def reset_state(self):
        self.state.zero_()

    def set_streaming(self, streaming=True):
        self.streaming = streaming


def streaming_normalized_conv1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(CausalConv1d(*args, **kwargs))


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        """
        Causal transposed convolution that also supports stateful buffers for streaming inference.

        usage:
            # Training loop (full sequences)
            model.train()
            model.conv.set_streaming(False)  # Disable streaming during training

            # Switch to streaming mode
            model.eval()
            model.conv.set_streaming(True)
            model.conv.reset_state()  # Reset state for a new sequence

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of the convolution.
            stride (int): Stride of the convolution.
            groups (int): Number of groups for grouped convolution.
            bias (bool): Whether to include a bias term in the convolution.
        """
        super().__init__()

        self.padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv_trans = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias
        )

        # State buffer for streaming (inference)
        self.register_buffer('state', torch.zeros(in_channels, self.padding))
        self.training_mode = True  # Default to training mode

    def forward(self, x):
        if self.streaming:
            # Use state for streaming inference
            x = torch.cat((self.state, x), dim=-1)
            out = self.conv(x)
            self.state = x[:, :, -self.padding:].detach()
            return out
        else:
            # Apply causal padding directly during training
            x_padded = F.pad(x, (self.padding, 0))
            return self.conv(x_padded)

    def reset_state(self):
        self.state.zero_()

    def set_streaming(self, streaming=True):
        self.streaming = streaming


def streaming_normalized_conv_trans1d(*args, **kwargs):
    return nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))
