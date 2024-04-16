import torch

class TimeMaskingBlockWaveform(torch.nn.Module):
    def __init__(self, masking_percentage=2):

        """
        Time masking in the time domain.

        Args:
            masking_percentage: percentage of time steps to mask
        """
        super().__init__()
        assert 0 <= masking_percentage <= 100, "masking_percentage should be in [0, 100]"

        self.masking_percentage = masking_percentage

    def forward(self, x):
        """

        Args:
            x(torch.Tensor): input waveform tensor of shape (..., time)

        Returns:
            torch.Tensor: masked tensor of shape (..., time)

        """

        time_samples = x.shape[-1]
        masked_samples = int(time_samples * self.masking_percentage / 100)
        first_masked_sample = torch.randint(0, time_samples - masked_samples, (1,)).item()

        # Masking
        x[..., first_masked_sample:first_masked_sample + masked_samples] = 0

        return x
